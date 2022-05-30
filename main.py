import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6'
import pickle
import trainer
import requests
import shutil

languages = ['EN', 'ES', 'PT']

print('\n .............  Downloading Multilingual Embedding ............. \n\n')
for lan in languages:
    filename = 'embedding1/wiki.%s.vec'%lan.lower()
    if not os.path.exists(filename):
        print('Downloading %s language vector'%lan)
        url = 'https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.%s.vec'%lan.lower()

        with requests.get(url, stream=True) as r:
            with open(filename, 'wb') as f:
                shutil.copyfileobj(r.raw, f)
        print('Done')


# It might take some time to prepare all the embeddings.
print('\n .............  Prepare Multilingual Embedding ............. \n\n')

from fasttext_multingual.fasttext import FastVector
dictionary={}

for lan in languages:
    dictionary[lan] = FastVector(vector_file='embedding/wiki.%s.vec'%lan.lower())
    dictionary[lan].apply_transform('fasttext_multingual/alignment_matrices/%s.txt'%lan.lower())




#  This is a demo that only trains 2 epochs  for training on EN,  ES , or EN+ES

method = 'pipeline'   # or  seperate
epochs = 2   # default is 200.
train_model_1 = True  # whether to train model 1 for classfication?  Select False for only evaluation
train_model_2 =  True  # whether to train model 2 for labeling?  Select False for only evaluation

for seed in [1]:   # select only one seed for demo.
    for lan in ['EN','ES','both']:
        print('\n\n\n\n      +++++++++++  seed:', seed, "language:", lan, "epochs:", epochs)
        record = trainer.run(seed,
                                  train_model_1=train_model_1,
                                  train_model_2=train_model_2,
                                  method=method,
                                  dictionary = dictionary,
                                  epochs=epochs,
                                  lan=lan)

        # Save your record to your favourite path.
        out_dir = 'multi/%s/%s_%s_%s_results.pickle' % (lan, lan, method, seed)
        print('\nsaving record to:\t %s'%out_dir)
        with open(out_dir, 'wb') as file:
            pickle.dump(record, file)




# #############  To reproduce our results in the paper  ##################
#
# method = 'pipeline'
# epochs = 200
# seeds = range(1, 11)  # There are totally 10 seeds' training/validation files.
# for seed in seeds:
#     for lan in ['EN','ES','both']:
#         print('\n\n\n\n      +++++++++++  seed:', seed, "language:", lan, "epochs:", epochs)
#         record = trainer.run(seed,
#                                   train_model_1=True,
#                                   train_model_2=True,
#                                   method='pipeline',
#                                   dictionary = dictionary,
#                                   epochs=200,
#                                   lan=lan)
#
#         # Save your record to your favourite path.
#         out_dir = 'multi/%s/%s_%s_%s_results.pickle' % (lan, lan, method, seed)
#         print('\nsaving record to:\t %s'%out_dir)
#         with open(out_dir, 'wb') as file:
#             pickle.dump(record, file)
#
# # Then we analysis all the saved records to get the average results.