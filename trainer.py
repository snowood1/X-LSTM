import time
import torch.optim as optim
import pandas as pd
import pickle
from models import ViterbiLoss, CRF, BiLSTM_FC, BiLSTM_CRF
from utils import *

from inference import ViterbiDecoder
from sklearn.metrics import f1_score
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from pathlib import Path

epochs = 200
method = 'seperate'

word_emb_dim = 300  # word embedding size
min_word_freq = 0  # threshold for word frequency
min_char_freq = 0  # threshold for character frequency
caseless = True  # lowercase everything?
expand_vocab = False  # expand model's input vocabulary to the pre-trained embeddings' vocabulary?
num_labels = 5

word_rnn_dim = 300  # word RNN size
word_rnn_layers = 1  # number of layers in word RNN
dropout = 0.7  # dropout
fine_tune_word_embeddings = False  # fine-tune pre-trained word embeddings?

# Training parameters
start_epoch = 0  # start at this epoch
batch_size = 16  # batch size
lr = 1e-3  # learning rate
lr_decay = 0.05  # decay learning rate by this amount
momentum = 0.9  # momentum
workers = 2 # number of workers for loading data in the DataLoader
grad_clip = 1.  # clip gradients at this value
print_freq = 20  # print training or validation status every __ batches
best_f1 = 0.  # F1 score to start with
checkpoint = None  # path to model checkpoint, None if none
# alpha = 1; beta = 0.5
languages = ['EN', 'ES', 'PT']
tag_ind = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def train_pentacode(train_loader, model, lm_criterion, optimizer, epoch):
    """
    Performs one epoch's training.

    :param train_loader: DataLoader for training data
    :param model: model
    :param lm_criterion: cross entropy loss layer
    :param crf_criterion: viterbi loss layer
    :param optimizer: optimizer
    :param epoch: epoch number
    :param vb_decoder: viterbi decoder (to decode and find F1 score)
    """

    model.train()  # training mode enables dropout

    data_time = AverageMeter()  # data loading time per batch
    ce_losses = AverageMeter()  # cross entropy loss
    accuracy = AverageMeter()

    # Batches
    for i, (wmaps, cmaps_f, cmaps_b, cmarkers_f, cmarkers_b, tmaps, wmap_lengths, cmap_lengths, labels) in enumerate(
            train_loader):
        max_word_len = max(wmap_lengths.tolist())
        # Reduce batch's padded length to maximum in-batch sequence
        # This saves some compute on nn.Linear layers (RNNs are unaffected, since they don't compute over the pads)
        wmaps = wmaps[:, :max_word_len].to(device)
        tmaps = tmaps[:, :max_word_len].to(device)
        wmap_lengths = wmap_lengths.to(device)
        labels = labels.to(device)

        logit = model(wmaps, wmap_lengths)

        # We don't predict the next word at the pads or <end> tokens
        # We will only predict at [dunston, checks, in] among [dunston, checks, in, <end>, <pad>, <pad>, ...]
        # So, prediction lengths are word sequence lengths - 1
        lm_lengths = wmap_lengths - 1
        lm_lengths = lm_lengths.tolist()

        pred = logit.data.max(1, keepdim=True)[1]
        correct = pred.eq(labels.data.view_as(pred)).sum().item()
        acc = correct / pred.size(0)

        ce_loss = lm_criterion(logit, labels)
        loss = ce_loss

        # Back prop.
        optimizer.zero_grad()
        loss.backward()

        if grad_clip is not None:
            clip_gradient(optimizer, grad_clip)

        optimizer.step()

        # Keep track of metrics

        ce_losses.update(ce_loss.item(), sum(lm_lengths))
        accuracy.update(acc, pred.size(0))
    if epoch % print_freq ==0:
        print(
        'Epoch: [{0}]\t LOSS - {ce_loss.avg:.3f}, acc - {acc.avg:.3f}'.format(epoch, ce_loss=ce_losses, acc=accuracy))


def validate_pentacode(val_loader, model, epoch):
    model.eval()
    accuracy = AverageMeter()
    for i, (wmaps, cmaps_f, cmaps_b, cmarkers_f, cmarkers_b, tmaps, wmap_lengths, cmap_lengths, labels) in enumerate(
            val_loader):
        max_word_len = max(wmap_lengths.tolist())

        # Reduce batch's padded length to maximum in-batch sequence
        # This saves some compute on nn.Linear layers (RNNs are unaffected, since they don't compute over the pads)
        wmaps = wmaps[:, :max_word_len].to(device)
        tmaps = tmaps[:, :max_word_len].to(device)
        wmap_lengths = wmap_lengths.to(device)
        labels = labels.to(device)

        # Forward prop.
        logit = model(wmaps, wmap_lengths)

        pred = logit.data.max(1, keepdim=True)[1]
        correct = pred.eq(labels.data.view_as(pred)).sum().item()
        acc = correct / pred.size(0)
        accuracy.update(acc, pred.size(0))
    if epoch % print_freq == 0:
        print(
            ' * LOSS - acc - {acc.avg:.3f}'.format(acc=accuracy))
    return accuracy.avg


def train_tagging(train_loader, model, crf_criterion, optimizer, epoch, vb_decoder, use_label_features=True):
    """
    Performs one epoch's training.

    :param train_loader: DataLoader for training data
    :param model: model
    :param lm_criterion: cross entropy loss layer
    :param crf_criterion: viterbi loss layer
    :param optimizer: optimizer
    :param epoch: epoch number
    :param vb_decoder: viterbi decoder (to decode and find F1 score)
    """

    model.train()  # training mode enables dropout

    batch_time = AverageMeter()  # forward prop. + back prop. time per batch
    data_time = AverageMeter()  # data loading time per batch
    ce_losses = AverageMeter()  # cross entropy loss
    vb_losses = AverageMeter()  # viterbi loss
    accuracy = AverageMeter()
    f1s = AverageMeter()  # f1 score

    start = time.time()

    # Batches
    for i, (wmaps, cmaps_f, cmaps_b, cmarkers_f, cmarkers_b, tmaps, wmap_lengths, cmap_lengths, labels) in enumerate(
            train_loader):
        data_time.update(time.time() - start)
        max_word_len = max(wmap_lengths.tolist())
        # Reduce batch's padded length to maximum in-batch sequence
        # This saves some compute on nn.Linear layers (RNNs are unaffected, since they don't compute over the pads)
        wmaps = wmaps[:, :max_word_len].to(device)
        tmaps = tmaps[:, :max_word_len].to(device)
        wmap_lengths = wmap_lengths.to(device)
        labels = labels.to(device) if use_label_features else None

        crf_scores, wmaps_sorted, tmaps_sorted, wmap_lengths_sorted, labels_sorted, _ = model(wmaps, tmaps,
                                                                                              wmap_lengths, labels)

        # We don't predict the next word at the pads or <end> tokens
        # We will only predict at [dunston, checks, in] among [dunston, checks, in, <end>, <pad>, <pad>, ...]
        # So, prediction lengths are word sequence lengths - 1
        lm_lengths = wmap_lengths_sorted - 1
        lm_lengths = lm_lengths.tolist()

        vb_loss = crf_criterion(crf_scores, tmaps_sorted, wmap_lengths_sorted)
        loss = vb_loss

        # Back prop.
        optimizer.zero_grad()
        loss.backward()

        if grad_clip is not None:
            clip_gradient(optimizer, grad_clip)
        optimizer.step()

        # Viterbi decode to find accuracy / f1
        decoded = vb_decoder.decode(crf_scores.to("cpu"), wmap_lengths_sorted.to("cpu"))

        # Remove timesteps we won't predict at, and also <end> tags, because to predict them would be cheating
        decoded, _, _, _ = pack_padded_sequence(decoded, lm_lengths, batch_first=True)
        tmaps_sorted = tmaps_sorted % vb_decoder.tagset_size  # actual target indices (see create_input_tensors())
        tmaps_sorted, _, _, _ = pack_padded_sequence(tmaps_sorted, lm_lengths, batch_first=True)

        # F1
        f1 = f1_score(tmaps_sorted.to("cpu").numpy(), decoded.numpy(), average='macro')

        # Keep track of metrics
        vb_losses.update(vb_loss.item(), crf_scores.size(0))
        batch_time.update(time.time() - start)
        f1s.update(f1, sum(lm_lengths))
    if epoch % print_freq ==0:
        print('Epoch: [{0}]\t LOSS - {vb_loss.avg:.3f}, F1 SCORE - {f1.avg:.3f}'.format(epoch, vb_loss=vb_losses, f1=f1s))


def validate_tagging(val_loader, model, crf_criterion, vb_decoder, epoch, use_label_features=True):
    """
    Performs one epoch's validation.

    :param val_loader: DataLoader for validation data
    :param model: model
    :param crf_criterion: viterbi loss layer
    :param vb_decoder: viterbi decoder
    :return: validation F1 score
    """
    model.eval()
    vb_losses = AverageMeter()
    f1s = AverageMeter()

    for i, (wmaps, cmaps_f, cmaps_b, cmarkers_f, cmarkers_b, tmaps, wmap_lengths, cmap_lengths, labels) in enumerate(
            val_loader):
        max_word_len = max(wmap_lengths.tolist())

        # Reduce batch's padded length to maximum in-batch sequence
        # This saves some compute on nn.Linear layers (RNNs are unaffected, since they don't compute over the pads)
        wmaps = wmaps[:, :max_word_len].to(device)
        tmaps = tmaps[:, :max_word_len].to(device)
        wmap_lengths = wmap_lengths.to(device)
        labels = labels.to(device) if use_label_features else None

        # Forward prop.
        crf_scores, wmaps_sorted, tmaps_sorted, wmap_lengths_sorted, labels_sorted, word_sort_ind \
            = model(wmaps, tmaps, wmap_lengths, labels)

        # Viterbi / CRF layer loss
        vb_loss = crf_criterion(crf_scores, tmaps_sorted, wmap_lengths_sorted)

        # Viterbi decode to find accuracy / f1
        decoded = vb_decoder.decode(crf_scores.to("cpu"), wmap_lengths_sorted.to("cpu"))

        # Remove timesteps we won't predict at, and also <end> tags, because to predict them would be cheating
        decoded, _, _, _ = pack_padded_sequence(decoded, (wmap_lengths_sorted - 1).tolist(), batch_first=True)
        tmaps_sorted = tmaps_sorted % vb_decoder.tagset_size  # actual target indices (see create_input_tensors())
        tmaps_sorted, _, _, _ = pack_padded_sequence(tmaps_sorted, (wmap_lengths_sorted - 1).tolist(), batch_first=True)

        # f1
        f1 = f1_score(tmaps_sorted.to("cpu").numpy(), decoded.numpy(), average='macro')

        # Keep track of metrics
        vb_losses.update(vb_loss.item(), crf_scores.size(0))
        f1s.update(f1, sum((wmap_lengths_sorted - 1).tolist()))
    if epoch % print_freq ==0:
        print(' * LOSS - {vb_loss.avg:.3f}, F1 SCORE - {f1.avg:.3f}'.format(vb_loss=vb_losses, f1=f1s))
    return f1s.avg


def eval_result(model2, word_map, tag_map, wmaps, tmaps, wmap_lengths, input_labels):
    model2.eval()

    vb_decoder = ViterbiDecoder(tag_map)

    crf_scores, wmaps_sorted, tmaps_sorted, wmap_lengths_sorted, labels_sorted, word_sort_ind \
        = model2(wmaps, tmaps, wmap_lengths, input_labels)  # label ==> pred

    # vb_loss = crf_criterion(crf_scores, tmaps_sorted, wmap_lengths_sorted)
    decoded = vb_decoder.decode(crf_scores.to("cpu"), wmap_lengths_sorted.to("cpu"))
    decoded1, _, _, _ = pack_padded_sequence(decoded, (wmap_lengths_sorted - 1).tolist(), batch_first=True)
    tmaps_sorted = tmaps_sorted % vb_decoder.tagset_size
    tmaps_sorted1, _, _, _ = pack_padded_sequence(tmaps_sorted, (wmap_lengths_sorted - 1).tolist(), batch_first=True)

    f1 = f1_score(tmaps_sorted1.to("cpu").numpy(), decoded1.numpy(), average='macro')
    print('f11', f1)

    df = pd.DataFrame([word_sort_ind.cpu().numpy(), wmaps_sorted.cpu().numpy(), \
                       decoded.cpu().numpy(), tmaps_sorted.cpu().numpy(), \
                       wmap_lengths_sorted.cpu().numpy(),
                       labels_sorted.cpu().numpy() if labels_sorted!=None else [None]*len(decoded)]
                      )
    df = df.T
    df.columns = ['id', 'word', 'tags', 'real_tags', 'length', 'input_label']
    df = df.sort_values('id').reset_index(drop=True)

    for i in range(df.shape[0]):
        length = df.length[i] - 1
        df.word[i] = df.word[i][0:length]
        df.tags[i] = df.tags[i][0:length]
        df.real_tags[i] = df.real_tags[i][0:length]

    new_word_map = {v: k for k, v in word_map.items()}
    new_tag_map = {v: k for k, v in tag_map.items()}
    new_tag_map = str(new_tag_map).replace('B-T', 'T').replace('I-T', 'T').replace('B-S', 'S').replace('I-S',
                                                                                                       'S').replace(
        'B-R', 'R').replace('I-R', 'R')
    new_tag_map = eval(new_tag_map)

    for i in range(df.shape[0]):
        df.word[i] = [new_word_map[i] for i in df.word[i]]
        df.tags[i] = [new_tag_map[i] for i in df.tags[i]]
        df.real_tags[i] = [new_tag_map[i] for i in df.real_tags[i]]

    return df


def get_results(word_map, tag_map, inputs, EVAL_TEST, method, models,seed):
    record={}
    wmaps, _, _, _, _, tmaps, wmap_lengths, _, labels = inputs
    max_word_len = max(wmap_lengths.tolist())
    wmaps = wmaps[:, :max_word_len].to(device)
    tmaps = tmaps[:, :max_word_len].to(device)
    wmap_lengths = wmap_lengths.to(device)
    labels = labels.to(device)

    if type(models)==list:
        model1, model2 = models[0],  models[1]

    ## model 1 ##

    print('\t====> Evaluating Model 1')
    logit = model1(wmaps, wmap_lengths)
    pred = logit.data.max(1)[1]
    correct = pred.eq(labels.data).sum().item()
    acc = correct / pred.size(0)
    print('%s accuracy' % EVAL_TEST, acc)

    record['%s_acc' % EVAL_TEST] = acc

    ## model 2 ##
    print('\n\t====> Evaluating Model 2')

    input_labels = None if method == 'seperate' else labels
    df = eval_result(model2, word_map, tag_map, wmaps, tmaps, wmap_lengths, input_labels)

    df.columns = ['id', 'word', 'tags', 'real_tags', 'length', 'label(gold)']

    ## TODO ###
    df['label(gold)'] = labels.cpu()
    df['label(pred)']= pred.cpu()

    # df_filename = '%s/%s/%s_gold_%s.csv' % (lan, method, EVAL_TEST, seed)
    # df.to_csv(df_filename, index=False)
    #
    # output_file = open('%s/%s/%s_gold_%s.txt' % (lan, method, EVAL_TEST, seed), "w")
    # for i in range(df.shape[0]):
    #     row = df.iloc[i, :]
    #     for word, tag in zip(row['word'], row['tags']):
    #         output_file.write("%s %s\n" % (word, tag))
    #     output_file.write('\n')
    # output_file.close()

    record['%s_gold' % EVAL_TEST] = df

    ## Pipeline  ##
    if method == 'pipeline':
        print('\n\t====> Evaluating Pipeline\n')
        df = eval_result(model2, word_map, tag_map, wmaps, tmaps, wmap_lengths, pred)
        df.columns = ['id', 'word', 'tags', 'real_tags', 'length', 'label(pred)']
        df['label(gold)'] = labels.cpu().numpy()

        # df.to_csv('pipeline_pred_eval_%s_%s.csv' % (lan, seed), index=False)
        # df_filename = '%s/%s/%s_pred_%s.csv' % (lan, method, EVAL_TEST, seed)
        # df.to_csv(df_filename, index=False)

        # output_file = open('%s/%s/%s_pred_%s.txt' % (lan, method, EVAL_TEST, seed), "w")
        # for i in range(df.shape[0]):
        #     row = df.iloc[i, :]
        #     # for j in range(row['length']-1):
        #     for word, tag in zip(row['word'], row['tags']):
        #         output_file.write("%s %s\n" % (word, tag))
        #     output_file.write('\n')
        # output_file.close()
        record['%s_pred' % EVAL_TEST] = df
    return record


def load_embeddings(fastvector, word_map, expand_vocab=True):

        emb_len = 300
        # print("Embedding length is %d." % emb_len)

        # Create tensor to hold embeddings for words that are in-corpus
        ic_embs = torch.FloatTensor(len(word_map), emb_len)
        init_embedding(ic_embs)

        if expand_vocab:
            print("You have elected to include embeddings that are out-of-corpus.")
            ooc_words = []
            ooc_embs = []
        else:
            print("You have elected NOT to include embeddings that are out-of-corpus.")

        # Read embedding file
        print("\nLoading embeddings...")

        # for line in fin:
        for emb_word, embedding in zip(fastvector.word2id, fastvector.embed):
            # line =  line.rstrip().split(' ')

            # emb_word = line[0]
            # embedding = list(map(lambda t: float(t), filter(lambda n: n and not n.isspace(), line[1:])))
            # embedding = list(map(float, line[1:]))

            if not expand_vocab and emb_word not in word_map:
                continue

            # If word is in train_vocab, store at the correct index (as in the word_map)
            if emb_word in word_map:
                ic_embs[word_map[emb_word]] = torch.FloatTensor(embedding)

            # If word is in dev or test vocab, store it and its embedding into lists
            elif expand_vocab:
                ooc_words.append(emb_word)
                ooc_embs.append(embedding)

        lm_vocab_size = len(word_map)  # keep track of lang. model's output vocab size (no out-of-corpus words)

        if expand_vocab:
            print("'word_map' is being updated accordingly.")
            for word in ooc_words:
                word_map[word] = len(word_map)
            ooc_embs = torch.FloatTensor(np.asarray(ooc_embs))
            embeddings = torch.cat([ic_embs, ooc_embs], 0)

        else:
            embeddings = ic_embs

        # Sanity check
        # assert embeddings.size(0) == len(word_map)   # TODO

        print("\nDone.\n Embedding vocabulary: %d\n Language Model vocabulary: %d.\n" % (len(word_map), lm_vocab_size))

        return embeddings, word_map, lm_vocab_size


def run_mono(seed, train_model_1, train_model_2, method, dictionary,lan, epochs=200):


    print('\n\n=============  Preparing all the multingual embedding: =====================')
    word_maps={}
    char_maps={}
    tag_maps={}
    embeddings={}
    lm_vocab_sizes={}
    temp_word_maps={}

    offset = 0
    for i in languages:
        print('\n\n=============  lanuage: %s ====================='%i)

        all_words, all_tags = [], []

        print('loading test file')
        test_ner = './datasets/ner/full_test_%s.txt' % i
        test_words, test_tags = read_words_tags(test_ner, tag_ind, caseless)

        all_words += test_words
        all_tags += test_tags

        try:
            print('loading train file')
            train_ner = './datasets/ner/seed_%s_train_%s.txt' % (seed, i)
            val_ner = './datasets/ner/seed_%s_dev_%s.txt' % (seed, i)

            train_words, train_tags = read_words_tags(train_ner, tag_ind, caseless)
            val_words, val_tags = read_words_tags(val_ner, tag_ind, caseless)

            all_words += train_words+val_words
            all_tags += train_tags + val_tags

        except:
            pass

        word_map, char_map, tag_map = create_maps(all_words, all_tags, min_word_freq,
                                                  min_char_freq)  # create word, char, tag maps

        emb_file = dictionary[i]
        embedding, word_map, lm_vocab_size = load_embeddings(emb_file, word_map,
                                                              expand_vocab=expand_vocab)  # load pre-trained embeddings

        # TODO: add offset
        if offset:
            for j in word_map:
                word_map[j] += offset
        offset += len(word_map)
        temp_word_map = {k: v for k, v in word_map.items() if v <= word_map['<unk>']}

        word_maps[i], char_maps[i], tag_maps[i], \
        embeddings[i], lm_vocab_sizes[i], temp_word_maps[i] =\
            word_map, char_map, tag_map, embedding, lm_vocab_size, temp_word_map


    multi_vocab_len = sum([len(i) for i in word_maps.values()])
    multi_lm_vocab_size = sum(list(lm_vocab_sizes.values()))
    multi_embeddings =  torch.cat(list(embeddings.values()))

    assert tag_maps['EN'] == tag_maps['ES'] == tag_maps['PT']


    test_inputs_all = {}
    for i in languages:
        # print('\n\n============= Test lanuage: %s ====================='%i)

        # print('loading test file')

        test_labels = pd.read_csv('./datasets/classification/full_test_%s.txt'%i, delimiter='\t', header=None)
        test_labels = test_labels.iloc[:,1].values
        test_ner = './datasets/ner/full_test_%s.txt' % i
        test_words, test_tags = read_words_tags(test_ner, tag_ind, caseless)

        test_inputs = create_input_tensors(test_words, test_tags, temp_word_maps[i], char_maps[i], tag_maps[i])
        test_inputs_all[i] = list(test_inputs) + [torch.tensor(test_labels)]


    # Train  Data

    print('\n\n============= Training data for lanuage: %s =====================' % lan)



    train_labels = pd.read_csv('./datasets/classification/seed_%s_train_%s.txt'%(seed, lan), delimiter='\t', header=None)
    train_labels = train_labels.iloc[:, 1].values
    train_ner = './datasets/ner/seed_%s_train_%s.txt' % (seed, lan)

    train_words, train_tags = read_words_tags(train_ner, tag_ind, caseless)
    train_inputs = create_input_tensors(train_words, train_tags, temp_word_maps[lan], char_maps[lan], tag_maps[lan])
    train_inputs = list(train_inputs) + [torch.tensor(train_labels)]


    val_labels = pd.read_csv('./datasets/classification/seed_%s_dev_%s.txt'%(seed, lan), delimiter='\t', header=None)
    val_labels = val_labels.iloc[:,1].values
    val_ner = './datasets/ner/seed_%s_dev_%s.txt' % (seed, lan)
    val_words, val_tags = read_words_tags(val_ner, tag_ind, caseless)
    val_inputs = create_input_tensors(val_words, val_tags, temp_word_maps[lan], char_maps[lan], tag_maps[lan])
    val_inputs = list(val_inputs) + [torch.tensor(val_labels)]


    # DataLoaders
    train_loader = torch.utils.data.DataLoader(WCDataset(*train_inputs), batch_size=batch_size, shuffle=True,
                                               num_workers=workers, pin_memory=False)
    val_loader = torch.utils.data.DataLoader(WCDataset(*val_inputs), batch_size=batch_size, shuffle=False,
                                             num_workers=workers, pin_memory=False)

    # Loss functions
    lm_criterion = nn.CrossEntropyLoss().to(device)
    crf_criterion = ViterbiLoss(tag_map).to(device)

    # Viterbi decoder (to find accuracy during validation)
    vb_decoder = ViterbiDecoder(tag_map)

    ## =========================  Models  ============================== ##
    model1_dir = 'multi/%s/pentacode'%lan
    model2_dir = 'multi/%s/%s'%(lan, method)

    Path(model1_dir).mkdir(parents=True, exist_ok=True)
    Path(model2_dir).mkdir(parents=True, exist_ok=True)

    model1_path = 'multi/%s/pentacode/%s.pth.tar'%(lan, seed)
    best_model1_path = model1_path+ '_BEST'

    model2_path = 'multi/%s/%s/%s.pth.tar'%(lan, method, seed)
    best_model2_path = model2_path+ '_BEST'

    print('model1_path:\t', model1_path)
    print('model2_path:\t', model2_path)



    if train_model_1:
        model1 = BiLSTM_FC(tagset_size=len(tag_map),
                           vocab_size= multi_vocab_len,
                           lm_vocab_size= multi_lm_vocab_size,
                           word_emb_dim=word_emb_dim,
                           word_rnn_dim=word_rnn_dim,
                           word_rnn_layers=word_rnn_layers,
                           dropout=dropout,
                           num_labels= num_labels).to(device)
        model1.init_word_embeddings(multi_embeddings.to(device))  # initialize embedding layer with pre-trained embeddings
        model1.fine_tune_word_embeddings(fine_tune_word_embeddings)  # fine-tune
        optimizer1 = optim.Adam(params=filter(lambda p: p.requires_grad, model1.parameters()), lr=lr)



        # Train Model 1 #
        print('\t====> Training Model 1\n')
        best_acc = 0
        for epoch in range(start_epoch, epochs):

            train_pentacode(train_loader=train_loader,
                  model=model1,
                  lm_criterion=lm_criterion,
                  optimizer=optimizer1,
                  epoch=epoch)

            val_acc = validate_pentacode(val_loader=val_loader,model=model1, epoch=epoch)

            # Did validation F1 score improve?
            is_best = val_acc > best_acc
            best_acc = max(val_acc, best_acc)
            if not is_best:
                epochs_since_improvement += 1
            else:
                epochs_since_improvement = 0

            # save_checkpoint(epoch, model1, optimizer1, val_acc, word_map, char_map, tag_map, lm_vocab_size, is_best, 'pentacode_%s_%s.pth.tar'%(lan,seed))
            # TODO ##########
            # save_checkpoint(epoch, model1, val_acc, is_best,
            #             '%s/pentacode_%s.pth.tar' % (lan, seed))

            save_checkpoint(epoch, model1, val_acc, is_best, model1_path)


    if train_model_2:
        print('\t====> Training Model 2\n')
        use_label_features = True if method == 'pipeline' else False
        # try:
        #     model2 = torch.load(best_model2_path)['model']
        #     print('successfully load the best model')
        # except:

        model2 = BiLSTM_CRF(tagset_size=len(tag_map),
                            vocab_size=multi_vocab_len,
                            lm_vocab_size=multi_lm_vocab_size,
                            word_emb_dim=word_emb_dim,
                            word_rnn_dim=word_rnn_dim,
                            word_rnn_layers=word_rnn_layers,
                            dropout=dropout,
                            num_labels=num_labels,
                            use_label_features=use_label_features  # TODO
                            ).to(device)
        model2.init_word_embeddings(multi_embeddings.to(device))  # initialize embedding layer with pre-trained embeddings
        model2.fine_tune_word_embeddings(fine_tune_word_embeddings)  # fine-tune
        optimizer2 = optim.Adam(params=filter(lambda p: p.requires_grad, model2.parameters()), lr=lr)

        # Train Model 2 #
        best_f1 = 0
        for epoch in range(start_epoch, epochs):

            # One epoch's training
            train_tagging(train_loader=train_loader,
                          model=model2,
                          crf_criterion=crf_criterion,
                          optimizer=optimizer2,
                          epoch=epoch,
                          vb_decoder=vb_decoder,
                          use_label_features=use_label_features)  # TODO

            # One epoch's validation
            val_f1 = validate_tagging(val_loader=val_loader,
                                      model=model2,
                                      crf_criterion=crf_criterion,
                                      vb_decoder=vb_decoder,
                                      epoch=epoch,
                                      use_label_features=use_label_features)

            # Did validation F1 score improve?
            is_best = val_f1 > best_f1
            best_f1 = max(val_f1, best_f1)
            if not is_best:
                epochs_since_improvement += 1
            else:
                epochs_since_improvement = 0

            save_checkpoint(epoch, model2, val_f1, is_best, model2_path)


    ## =========================     Evaluation    =========================== ##

    print('\n\n ===== 3 Evaluating ====== \n')

    model1 = torch.load(best_model1_path)['model']
    model2 = torch.load(best_model2_path)['model']

    model1.init_word_embeddings(multi_embeddings.to(device))
    model2.init_word_embeddings(multi_embeddings.to(device))


    record= []

    val_record = get_results(word_maps[lan], tag_map, val_inputs, 'valid', method, [model1, model2], seed)

    for i in ['EN', 'ES', 'PT']:
        print('\n\n============= Test lanuage: %s ====================='%i)

        test_record = get_results(word_maps[i], tag_map, test_inputs_all[i], 'test', method, [model1, model2], seed)

        test_record['method'] = method
        test_record['seed'] = seed
        test_record['train_lan'] = lan
        test_record['test_lan'] = i

        record.append(test_record)
    return record


def run_multi(seed, train_model_1, train_model_2, method, dictionary,lan='both', epochs=200):

    print('\n\n=============  Preparing all the multingual embedding: =====================')
    word_maps={}
    char_maps={}
    tag_maps={}
    embeddings={}
    lm_vocab_sizes={}
    temp_word_maps={}

    offset = 0
    for i in languages:
        print('\n\n=============  lanuage: %s ====================='%i)

        all_words, all_tags = [], []

        print('loading test file')
        test_ner = './datasets/ner/full_test_%s.txt' % i
        test_words, test_tags = read_words_tags(test_ner, tag_ind, caseless)

        all_words += test_words
        all_tags += test_tags

        try:
            print('loading train file')
            train_ner = './datasets/ner/seed_%s_train_%s.txt' % (seed, i)
            val_ner = './datasets/ner/seed_%s_dev_%s.txt' % (seed, i)

            train_words, train_tags = read_words_tags(train_ner, tag_ind, caseless)
            val_words, val_tags = read_words_tags(val_ner, tag_ind, caseless)

            all_words += train_words+val_words
            all_tags += train_tags + val_tags

        except:
            pass

        word_map, char_map, tag_map = create_maps(all_words, all_tags, min_word_freq,
                                                  min_char_freq)  # create word, char, tag maps

        emb_file = dictionary[i]
        embedding, word_map, lm_vocab_size = load_embeddings(emb_file, word_map,
                                                              expand_vocab=expand_vocab)  # load pre-trained embeddings

        # TODO: add offset
        if offset:
            for j in word_map:
                word_map[j] += offset
        offset += len(word_map)
        temp_word_map = {k: v for k, v in word_map.items() if v <= word_map['<unk>']}

        word_maps[i], char_maps[i], tag_maps[i], \
        embeddings[i], lm_vocab_sizes[i], temp_word_maps[i] =\
            word_map, char_map, tag_map, embedding, lm_vocab_size, temp_word_map


    multi_vocab_len = sum([len(i) for i in word_maps.values()])
    multi_lm_vocab_size = sum(list(lm_vocab_sizes.values()))
    multi_embeddings =  torch.cat(list(embeddings.values()))

    assert tag_maps['EN'] == tag_maps['ES'] == tag_maps['PT']


    test_inputs_all = {}
    for i in languages:
        # print('\n\n============= Test lanuage: %s ====================='%i)

        # print('loading test file')

        test_labels = pd.read_csv('./datasets/classification/full_test_%s.txt'%i, delimiter='\t', header=None)
        test_labels = test_labels.iloc[:,1].values
        test_ner = './datasets/ner/full_test_%s.txt' % i
        test_words, test_tags = read_words_tags(test_ner, tag_ind, caseless)

        test_inputs = create_input_tensors(test_words, test_tags, temp_word_maps[i], char_maps[i], tag_maps[i])
        test_inputs_all[i] = list(test_inputs) + [torch.tensor(test_labels)]


    # Train  Data

    print('\n\n============= Training data for lanuage: EN + ES =====================')
    assert lan =='both'

    lan ='EN'

    train_labels = pd.read_csv('./datasets/classification/seed_%s_train_%s.txt'%(seed, lan), delimiter='\t', header=None)
    train_labels = train_labels.iloc[:, 1].values
    train_ner = './datasets/ner/seed_%s_train_%s.txt' % (seed, lan)

    train_words, train_tags = read_words_tags(train_ner, tag_ind, caseless)
    train_inputs = create_input_tensors(train_words, train_tags, temp_word_maps[lan], char_maps[lan], tag_maps[lan])
    train_inputs = list(train_inputs) + [torch.tensor(train_labels)]


    val_labels = pd.read_csv('./datasets/classification/seed_%s_dev_%s.txt'%(seed, lan), delimiter='\t', header=None)
    val_labels = val_labels.iloc[:,1].values
    val_ner = './datasets/ner/seed_%s_dev_%s.txt' % (seed, lan)
    val_words, val_tags = read_words_tags(val_ner, tag_ind, caseless)
    val_inputs = create_input_tensors(val_words, val_tags, temp_word_maps[lan], char_maps[lan], tag_maps[lan])
    val_inputs = list(val_inputs) + [torch.tensor(val_labels)]

    # DataLoaders
    train_loader1 = torch.utils.data.DataLoader(WCDataset(*train_inputs), batch_size=batch_size, shuffle=True,
                                               num_workers=workers, pin_memory=False)
    val_loader1 = torch.utils.data.DataLoader(WCDataset(*val_inputs), batch_size=batch_size, shuffle=False,
                                             num_workers=workers, pin_memory=False)

    lan ='ES'

    train_labels = pd.read_csv('./datasets/classification/seed_%s_train_%s.txt' % (seed, lan), delimiter='\t',
                               header=None)
    train_labels = train_labels.iloc[:, 1].values
    train_ner = './datasets/ner/seed_%s_train_%s.txt' % (seed, lan)

    train_words, train_tags = read_words_tags(train_ner, tag_ind, caseless)
    train_inputs = create_input_tensors(train_words, train_tags, temp_word_maps[lan], char_maps[lan], tag_maps[lan])
    train_inputs = list(train_inputs) + [torch.tensor(train_labels)]

    val_labels = pd.read_csv('./datasets/classification/seed_%s_dev_%s.txt' % (seed, lan), delimiter='\t', header=None)
    val_labels = val_labels.iloc[:, 1].values
    val_ner = './datasets/ner/seed_%s_dev_%s.txt' % (seed, lan)
    val_words, val_tags = read_words_tags(val_ner, tag_ind, caseless)
    val_inputs = create_input_tensors(val_words, val_tags, temp_word_maps[lan], char_maps[lan], tag_maps[lan])
    val_inputs = list(val_inputs) + [torch.tensor(val_labels)]

    # DataLoaders
    train_loader2 = torch.utils.data.DataLoader(WCDataset(*train_inputs), batch_size=batch_size, shuffle=True,
                                                num_workers=workers, pin_memory=False)
    val_loader2 = torch.utils.data.DataLoader(WCDataset(*val_inputs), batch_size=batch_size, shuffle=False,
                                              num_workers=workers, pin_memory=False)


    lan = 'both'

    # Loss functions
    lm_criterion = nn.CrossEntropyLoss().to(device)
    crf_criterion = ViterbiLoss(tag_map).to(device)

    # Viterbi decoder (to find accuracy during validation)
    vb_decoder = ViterbiDecoder(tag_map)

    ## =========================  Models  ============================== ##
    model1_dir = 'multi/%s/pentacode'%lan
    model2_dir = 'multi/%s/%s'%(lan, method)

    Path(model1_dir).mkdir(parents=True, exist_ok=True)
    Path(model2_dir).mkdir(parents=True, exist_ok=True)

    model1_path = 'multi/%s/pentacode/%s.pth.tar'%(lan, seed)
    best_model1_path = model1_path+ '_BEST'

    model2_path = 'multi/%s/%s/%s.pth.tar'%(lan, method, seed)
    best_model2_path = model2_path+ '_BEST'

    print('model1_path:\t', model1_path)
    print('model2_path:\t', model2_path)



    if train_model_1:
        model1 = BiLSTM_FC(tagset_size=len(tag_map),
                           vocab_size= multi_vocab_len,
                           lm_vocab_size= multi_lm_vocab_size,
                           word_emb_dim=word_emb_dim,
                           word_rnn_dim=word_rnn_dim,
                           word_rnn_layers=word_rnn_layers,
                           dropout=dropout,
                           num_labels= num_labels).to(device)
        model1.init_word_embeddings(multi_embeddings.to(device))  # initialize embedding layer with pre-trained embeddings
        model1.fine_tune_word_embeddings(fine_tune_word_embeddings)  # fine-tune
        optimizer1 = optim.Adam(params=filter(lambda p: p.requires_grad, model1.parameters()), lr=lr)



        # Train Model 1 #
        print('\t====> Training Model 1\n')
        best_acc = 0
        for epoch in range(start_epoch, epochs):

            train_pentacode(train_loader=train_loader1,
                  model=model1,
                  lm_criterion=lm_criterion,
                  optimizer=optimizer1,
                  epoch=epoch)

            train_pentacode(train_loader=train_loader2,
                  model=model1,
                  lm_criterion=lm_criterion,
                  optimizer=optimizer1,
                  epoch=epoch)

            val_acc1 = validate_pentacode(val_loader=val_loader1,model=model1, epoch=epoch)
            val_acc2 = validate_pentacode(val_loader=val_loader2,model=model1, epoch=epoch)

            val_acc = (val_acc1 + val_acc2)/2

            # Did validation F1 score improve?
            is_best = val_acc > best_acc
            best_acc = max(val_acc, best_acc)
            if not is_best:
                epochs_since_improvement += 1
            else:
                epochs_since_improvement = 0

            # save_checkpoint(epoch, model1, optimizer1, val_acc, word_map, char_map, tag_map, lm_vocab_size, is_best, 'pentacode_%s_%s.pth.tar'%(lan,seed))
            # TODO ##########
            # save_checkpoint(epoch, model1, val_acc, is_best,
            #             '%s/pentacode_%s.pth.tar' % (lan, seed))

            save_checkpoint(epoch, model1, val_acc, is_best, model1_path)


    if train_model_2:
        print('\t====> Training Model 2\n')
        use_label_features = True if method == 'pipeline' else False
        # try:
        #     model2 = torch.load(best_model2_path)['model']
        #     print('successfully load the best model')
        # except:

        model2 = BiLSTM_CRF(tagset_size=len(tag_map),
                            vocab_size=multi_vocab_len,
                            lm_vocab_size=multi_lm_vocab_size,
                            word_emb_dim=word_emb_dim,
                            word_rnn_dim=word_rnn_dim,
                            word_rnn_layers=word_rnn_layers,
                            dropout=dropout,
                            num_labels=num_labels,
                            use_label_features=use_label_features  # TODO
                            ).to(device)
        model2.init_word_embeddings(multi_embeddings.to(device))  # initialize embedding layer with pre-trained embeddings
        model2.fine_tune_word_embeddings(fine_tune_word_embeddings)  # fine-tune
        optimizer2 = optim.Adam(params=filter(lambda p: p.requires_grad, model2.parameters()), lr=lr)

        # Train Model 2 #
        best_f1 = 0
        for epoch in range(start_epoch, epochs):

            # One epoch's training
            train_tagging(train_loader=train_loader1,
                          model=model2,
                          crf_criterion=crf_criterion,
                          optimizer=optimizer2,
                          epoch=epoch,
                          vb_decoder=vb_decoder,
                          use_label_features=use_label_features)  # TODO


            # One epoch's training
            train_tagging(train_loader=train_loader2,
                          model=model2,
                          crf_criterion=crf_criterion,
                          optimizer=optimizer2,
                          epoch=epoch,
                          vb_decoder=vb_decoder,
                          use_label_features=use_label_features)  # TODO

            # One epoch's validation
            val_f11 = validate_tagging(val_loader=val_loader1,
                                      model=model2,
                                      crf_criterion=crf_criterion,
                                      vb_decoder=vb_decoder,
                                      epoch=epoch,
                                      use_label_features=use_label_features)

            # One epoch's validation
            val_f12 = validate_tagging(val_loader=val_loader2,
                                      model=model2,
                                      crf_criterion=crf_criterion,
                                      vb_decoder=vb_decoder,
                                      epoch=epoch,
                                      use_label_features=use_label_features)

            val_f1 = (val_f11 + val_f12)/2


            # Did validation F1 score improve?
            is_best = val_f1 > best_f1
            best_f1 = max(val_f1, best_f1)
            if not is_best:
                epochs_since_improvement += 1
            else:
                epochs_since_improvement = 0

            save_checkpoint(epoch, model2, val_f1, is_best, model2_path)


    ## =========================     Evaluation    =========================== ##

    print('\n\n ===== 3 Evaluating ====== \n')

    model1 = torch.load(best_model1_path)['model']
    model2 = torch.load(best_model2_path)['model']

    model1.init_word_embeddings(multi_embeddings.to(device))
    model2.init_word_embeddings(multi_embeddings.to(device))


    record= []
    #
    # val_record = get_results(word_maps[lan], tag_map, val_inputs, 'valid', method, [model1, model2], seed)
    # val_record2 = get_results(word_maps[lan], tag_map, val_inputs, 'valid', method, [model1, model2], seed)

    for i in ['EN', 'ES', 'PT']:
        print('\n\n============= Test lanuage: %s ====================='%i)

        test_record = get_results(word_maps[i], tag_map, test_inputs_all[i], 'test', method, [model1, model2], seed)

        test_record['method'] = method
        test_record['seed'] = seed
        test_record['train_lan'] = lan
        test_record['test_lan'] = i

        record.append(test_record)
    return record


def run(seed, train_model_1, train_model_2, method, dictionary,lan, epochs=200):
    if lan == 'both':
        record = run_multi(seed, train_model_1, train_model_2, method, dictionary,lan, epochs)
    else:
        record = run_mono(seed, train_model_1, train_model_2, method, dictionary, lan, epochs)
    return record