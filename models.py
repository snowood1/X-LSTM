import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CRF(nn.Module):
    """
    Conditional Random Field.
    """

    def __init__(self, hidden_dim, tagset_size):
        """
        :param hidden_dim: size of word RNN/BLSTM's output
        :param tagset_size: number of tags
        """
        super(CRF, self).__init__()
        self.tagset_size = tagset_size
        self.emission = nn.Linear(hidden_dim, self.tagset_size)
        self.transition = nn.Parameter(torch.Tensor(self.tagset_size, self.tagset_size))
        self.transition.data.zero_()

    def forward(self, feats):
        """
        Forward propagation.

        :param feats: output of word RNN/BLSTM, a tensor of dimensions (batch_size, timesteps, hidden_dim)
        :return: CRF scores, a tensor of dimensions (batch_size, timesteps, tagset_size, tagset_size)
        """
        self.batch_size = feats.size(0)
        self.timesteps = feats.size(1)

        emission_scores = self.emission(feats)  # (batch_size, timesteps, tagset_size)
        emission_scores = emission_scores.unsqueeze(2).expand(self.batch_size, self.timesteps, self.tagset_size,
                                                              self.tagset_size)  # (batch_size, timesteps, tagset_size, tagset_size)

        crf_scores = emission_scores + self.transition.unsqueeze(0).unsqueeze(
            0)  # (batch_size, timesteps, tagset_size, tagset_size)
        return crf_scores


class ViterbiLoss(nn.Module):
    """
    Viterbi Loss.
    """

    def __init__(self, tag_map):
        """
        :param tag_map: tag map
        """
        super(ViterbiLoss, self).__init__()
        self.tagset_size = len(tag_map)
        self.start_tag = tag_map['<start>']
        self.end_tag = tag_map['<end>']

    def forward(self, scores, targets, lengths):
        """
        Forward propagation.

        :param scores: CRF scores
        :param targets: true tags indices in unrolled CRF scores
        :param lengths: word sequence lengths
        :return: viterbi loss
        """

        batch_size = scores.size(0)
        word_pad_len = scores.size(1)

        # Gold score

        targets = targets.unsqueeze(2)
        scores_at_targets = torch.gather(scores.view(batch_size, word_pad_len, -1), 2, targets).squeeze(
            2)  # (batch_size, word_pad_len)

        # Everything is already sorted by lengths
        scores_at_targets, _,_,_ = pack_padded_sequence(scores_at_targets, lengths.cpu(), batch_first=True)
        gold_score = scores_at_targets.sum()

        # All paths' scores

        # Create a tensor to hold accumulated sequence scores at each current tag
        scores_upto_t = torch.zeros(batch_size, self.tagset_size).to(device)

        for t in range(max(lengths)):
            batch_size_t = sum([l > t for l in lengths])  # effective batch size (sans pads) at this timestep
            if t == 0:
                scores_upto_t[:batch_size_t] = scores[:batch_size_t, t, self.start_tag, :]  # (batch_size, tagset_size)
            else:
                # We add scores at current timestep to scores accumulated up to previous timestep, and log-sum-exp
                # Remember, the cur_tag of the previous timestep is the prev_tag of this timestep
                # So, broadcast prev. timestep's cur_tag scores along cur. timestep's cur_tag dimension
                scores_upto_t[:batch_size_t] = log_sum_exp(
                    scores[:batch_size_t, t, :, :] + scores_upto_t[:batch_size_t].unsqueeze(2),
                    dim=1)  # (batch_size, tagset_size)

        # We only need the final accumulated scores at the <end> tag
        all_paths_scores = scores_upto_t[:, self.end_tag].sum()

        viterbi_loss = all_paths_scores - gold_score
        viterbi_loss = viterbi_loss / batch_size

        return viterbi_loss


class BiLSTM_FC(nn.Module):

    def __init__(self, tagset_size, vocab_size,
                 lm_vocab_size, word_emb_dim, word_rnn_dim, word_rnn_layers, dropout,num_labels):
        """
        :param tagset_size: number of tags
        :param vocab_size: input vocabulary size
        :param lm_vocab_size: vocabulary size of language models (in-corpus words subject to word frequency threshold)
        :param word_emb_dim: size of word embeddings
        :param word_rnn_dim: size of word RNN/BLSTM
        :param word_rnn_layers:  number of layers in word RNNs/LSTMs
        :param dropout: dropout
        """

        super(BiLSTM_FC, self).__init__()

        self.tagset_size = tagset_size  # this is the size of the output vocab of the tagging model
        self.wordset_size = vocab_size  # this is the size of the input vocab (embedding layer) of the tagging model
        self.lm_vocab_size = lm_vocab_size  # this is the size of the output vocab of the language model
        self.word_emb_dim = word_emb_dim
        self.word_rnn_dim = word_rnn_dim
        self.word_rnn_layers = word_rnn_layers

        self.dropout = nn.Dropout(p=dropout)

        self.word_embeds = nn.Embedding(self.wordset_size, self.word_emb_dim)  # word embedding layer
        self.word_blstm = nn.LSTM(self.word_emb_dim, self.word_rnn_dim // 2,
                                  num_layers=self.word_rnn_layers, bidirectional=True, dropout=dropout)  # word BLSTM
        self.fc1 = nn.Linear(word_rnn_dim, word_rnn_dim)
        self.fc2 = nn.Linear(word_rnn_dim, num_labels)

    def init_word_embeddings(self, embeddings):
        """
        Initialize embeddings with pre-trained embeddings.

        :param embeddings: pre-trained embeddings
        """
        self.word_embeds.weight = nn.Parameter(embeddings)

    def fine_tune_word_embeddings(self, fine_tune=False):
        """
        Fine-tune embedding layer? (Not fine-tuning only makes sense if using pre-trained embeddings).

        :param fine_tune: Fine-tune?
        """
        for p in self.word_embeds.parameters():
            p.requires_grad = fine_tune

    def forward(self, wmaps, wmap_lengths):
        """
        Forward propagation.

        :param wmaps: padded encoded word sequences, a tensor of dimensions (batch_size, word_pad_len)
        :param tmaps: padded tag sequences, a tensor of dimensions (batch_size, word_pad_len)
        :param wmap_lengths: word sequence lengths, a tensor of dimensions (batch_size)
        :param cmap_lengths: character sequence lengths, a tensor of dimensions (batch_size, word_pad_len)
        """
        self.batch_size = wmaps.size(0)
        self.word_pad_len = wmaps.size(1)

        # # Sort by decreasing true word sequence length
        wmap_lengths, word_sort_ind = wmap_lengths.sort(dim=0, descending=True)

        # Embedding look-up for words
        w = self.word_embeds(wmaps)  # (batch_size, word_pad_len, word_emb_dim)
        w = self.dropout(w)

        # Pack padded sequence
        w = pack_padded_sequence(w, list(wmap_lengths),
                                 batch_first=True)  # packed sequence of word_emb_dim + 2 * char_rnn_dim, with real sequence lengths
        # LSTM
        w, (hidden, cell) = self.word_blstm(w)  # packed sequence of word_rnn_dim, with real sequence lengths

        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)

        logit = self.fc1(hidden)
        logit = F.relu(logit)
        logit = self.fc2(logit)

        return logit


class BiLSTM_CRF(nn.Module):
    """
    The encompassing LM-LSTM-CRF model.
    """

    def __init__(self, tagset_size, vocab_size,
                 lm_vocab_size, word_emb_dim, word_rnn_dim, word_rnn_layers, dropout, num_labels, use_label_features=True):
        """
        :param tagset_size: number of tags
        :param vocab_size: input vocabulary size
        :param lm_vocab_size: vocabulary size of language models (in-corpus words subject to word frequency threshold)
        :param word_emb_dim: size of word embeddings
        :param word_rnn_dim: size of word RNN/BLSTM
        :param word_rnn_layers:  number of layers in word RNNs/LSTMs
        :param dropout: dropout
        """

        super(BiLSTM_CRF, self).__init__()

        self.tagset_size = tagset_size  # this is the size of the output vocab of the tagging model
        self.wordset_size = vocab_size  # this is the size of the input vocab (embedding layer) of the tagging model
        self.lm_vocab_size = lm_vocab_size  # this is the size of the output vocab of the language model
        self.word_emb_dim = word_emb_dim
        self.word_rnn_dim = word_rnn_dim
        self.word_rnn_layers = word_rnn_layers

        self.label_embeds = nn.Embedding(num_labels, 1)

        self.dropout = nn.Dropout(p=dropout)

        self.word_embeds = nn.Embedding(self.wordset_size, self.word_emb_dim)  # word embedding layer
        self.word_blstm = nn.LSTM(self.word_emb_dim + (1 if use_label_features else 0), self.word_rnn_dim // 2,
                                  num_layers=self.word_rnn_layers, bidirectional=True, dropout=dropout)  # word BLSTM
        self.crf = CRF((self.word_rnn_dim // 2) * 2, self.tagset_size)  # conditional random field

    def init_word_embeddings(self, embeddings):
        """
        Initialize embeddings with pre-trained embeddings.

        :param embeddings: pre-trained embeddings
        """
        self.word_embeds.weight = nn.Parameter(embeddings)

    def fine_tune_word_embeddings(self, fine_tune=False):
        """
        Fine-tune embedding layer? (Not fine-tuning only makes sense if using pre-trained embeddings).

        :param fine_tune: Fine-tune?
        """
        for p in self.word_embeds.parameters():
            p.requires_grad = fine_tune

    def forward(self, wmaps, tmaps, wmap_lengths, labels):
        """
        Forward propagation.

        :param wmaps: padded encoded word sequences, a tensor of dimensions (batch_size, word_pad_len)
        :param tmaps: padded tag sequences, a tensor of dimensions (batch_size, word_pad_len)
        :param wmap_lengths: word sequence lengths, a tensor of dimensions (batch_size)
        :param cmap_lengths: character sequence lengths, a tensor of dimensions (batch_size, word_pad_len)
        """
        self.batch_size = wmaps.size(0)
        self.word_pad_len = wmaps.size(1)

        # # Sort by decreasing true word sequence length
        wmap_lengths, word_sort_ind = wmap_lengths.sort(dim=0, descending=True)
        wmaps = wmaps[word_sort_ind]
        tmaps = tmaps[word_sort_ind]

        # Embedding look-up for words
        w = self.word_embeds(wmaps)  # (batch_size, word_pad_len, word_emb_dim)
        w = self.dropout(w)


        # Sub-word information at each word
        # print(labels.shape)

        if labels != None:
            labels = labels[word_sort_ind]
            pentacodes = self.label_embeds(labels.view(-1, 1).repeat(1, self.word_pad_len))
            pentacodes = self.dropout(pentacodes)

             # Concatenate word embeddings and sub-word features
            w = torch.cat((w, pentacodes), dim=2)  # (batch_size, word_pad_len, word_emb_dim + 2 * char_rnn_dim)


        # Pack padded sequence
        w = pack_padded_sequence(w, list(wmap_lengths),
                                 batch_first=True)  # packed sequence of word_emb_dim + 2 * char_rnn_dim, with real sequence lengths
        # LSTM
        w, (hidden, cell) = self.word_blstm(w)  # packed sequence of word_rnn_dim, with real sequence lengths

        # Unpack packed sequence
        w, _ = pad_packed_sequence(w, batch_first=True)  # (batch_size, max_word_len_in_batch, word_rnn_dim)
        w = self.dropout(w)
        crf_scores = self.crf(w)  # (batch_size, max_word_len_in_batch, tagset_size, tagset_size)

        return crf_scores, wmaps, tmaps, wmap_lengths, labels, word_sort_ind
