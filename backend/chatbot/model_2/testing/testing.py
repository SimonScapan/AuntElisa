#Import packages
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os
import pickle
import random
import re
import statistics
import string
import sys
import tensorflow as tf
import time

#Import functions and classes
from nltk.translate.bleu_score import corpus_bleu
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import add
from tensorflow.keras.layers import Input, Dense, LSTM, GRU, TimeDistributed
from tensorflow.keras.layers import Embedding, Dropout, Bidirectional, Concatenate, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import add, Input, Dense, LSTM, GRU, TimeDistributed, Embedding, Dropout, Bidirectional, Concatenate, Lambda

#Add the parent directory to sys.path
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

#Import own functions
from functions import split_dataset, make_embedding_layer, answer

#Import own classes
from classes import Encoder, BahdanauAttention, Decoder

np.random.seed(1)
random.seed(1)

#Define some hyperparameters
test = False
print('test: ', test)
if test:
    GRU_units = 10
    batch_size = 4
    emb_dim = 10
else:
    GRU_units = 256
    batch_size = 32
    emb_dim = 50

init_lr = 0.0005

#Load the preprocessed data
with open('../preprocessing/preprocessed_data.pkl', 'rb') as f:
        preprocessed_data = pickle.load(f)

wordtoix = preprocessed_data['word2ix']
ixtoword = preprocessed_data['ixtoword']
pairs_final = preprocessed_data['pairs_final_train']
short_vocab = preprocessed_data['short_vocab']

max_len_q = preprocessed_data['max_len_q']
max_len_a = preprocessed_data['max_len_q']

end_token = '<EOS>'
start_token = '<BOS>'
pad_token = 'pad0'

ixtoword[0] = pad_token

#Splits the data into train (70%), test (15%) and valid(15%)
pairs_final_train, pairs_final_test, pairs_final_valid = split_dataset(pairs_final)

#Since index 0 is used as padding, we have to increase the vocab size
vocab_len = len(short_vocab) + 2

#Making the embedding mtrix and decide whether to use pretrained word embeddings
embeddings = make_embedding_layer(vocab_len=vocab_len, wordtoix=wordtoix, embedding_dim=emb_dim, glove=False)

#Create encoder
encoder = Encoder(vocab_len, emb_dim, GRU_units, batch_size, max_len_q, embeddings)

#Define the decoder of the network
decoder = Decoder(vocab_len, emb_dim, GRU_units, batch_size, embeddings)

#Applying the adam optimizer to update the network weights
optimizer = tf.keras.optimizers.Adam(init_lr)

#Initalize a checkpoint to save the model later
checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)

#Define and calculate hyperparameters for training/testing
history={'TestSimilarity':[]}
smallest_loss = np.inf
best_ep = 1
enc_hidden = encoder.initialize_hidden_state()

#Define tested epochs
start_epoch = 0
number_of_epochs = 200
step = 20

#Define testing
GRU_units = 10
batch_size = 4
training = False
test = True

with open('test_history.pkl', 'rb') as f:
  history = pickle.load(f)

for tested_epoch in range(start_epoch, number_of_epochs+1, step):

  if tested_epoch == 0:
    tested_epoch = 1

  #Load the model to be tested
  checkpoint.restore("../training/model/ckpt-" + str(tested_epoch))
  similarities = []
  print('Testing epoch ', tested_epoch)

  for test_record in pairs_final_test:
      #Define question, real answer and predicted answer
      question = test_record[0]
      label = test_record[1]
      output = answer(question, max_len_a, max_len_q, wordtoix, start_token, end_token, GRU_units, encoder, decoder, ixtoword)[:-1]

      #Split words of the sentences 
      label_list = label.split(' ')
      output_list = output.split(' ')

      #Remove stop words from the string 
      label_set = set(label_list)  
      output_set = set(output_list)

      #Create empty word vectors
      l1 =[];l2 =[]

      #Form a set containing keywords of both strings  
      rvector = label_set.union(output_set)  
      for w in rvector: 
          if w in label_set: l1.append(1) # create a vector 
          else: l1.append(0) 
          if w in output_set: l2.append(1) 
          else: l2.append(0) 
      c = 0
      
      #Compute the cosine similarity  
      for i in range(len(rvector)): 
              c+= l1[i]*l2[i]
      cosine = c / float((sum(l1)*sum(l2))**0.5) 
      similarities.append(cosine)

  #Compute average similarity per epoch
  similarity = statistics.mean(similarities)
  print(f'Similarity of epoche {tested_epoch}: {similarity}')
  print('---------------------------------------------------------')

  #Save the tested results
  history['TestSimilarity'].append(similarity)

  with open('test_history.pkl', 'wb') as f:
      pickle.dump(history, f)