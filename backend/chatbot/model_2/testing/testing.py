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
from functions import clean_text, split_dataset, progressBar, make_embedding_layer, evaluate, plot_attention, answer, loss_function, test_bot, plot_history

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
embeddings = make_embedding_layer(vocab_len=vocab_len, wordtoix=wordtoix, embedding_dim=emb_dim, glove=not test)

#Create encoder
encoder = Encoder(vocab_len, emb_dim, GRU_units, batch_size, max_len_q, embeddings)

#Define the decoder of the network
decoder = Decoder(vocab_len, emb_dim, GRU_units, batch_size, embeddings)

#Applying the adam optimizer to update the network weights
optimizer = tf.keras.optimizers.Adam(init_lr)

#Initalize a checkpoint to save the model later
checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)
manager = tf.train.CheckpointManager(checkpoint, '../training/model', max_to_keep = 300)

#Define and calculate hyperparameters for training/testing
history={'loss':[], 'lossTest':[]}
smallest_loss = np.inf
best_ep = 1
EPOCHS = 201 # but 150 is enough
enc_hidden = encoder.initialize_hidden_state()
steps_per_epoch = len(pairs_final_train)//batch_size # used for caculating number of batches
current_ep = 1
batch_per_epoche = 6

#Load the model to be tested
checkpoint.restore(manager.latest_checkpoint)

print("Restored from {}".format(manager.latest_checkpoint))
tested_epoch = int(manager.latest_checkpoint[36:])

with open('../training/training_history.pkl', 'rb') as f:
  history = pickle.load(f)

batch_loss = K.constant(0)
X, y = [], []


#Test the loaded epoche based on test data
GRU_units = 10
pairs_final_test_new = []
training = False
test = True
test_progress = 0

for test_record in pairs_final_test:
    test_progress+=1
    print('Testing this epoche')
    progressBar(value=test_progress,endvalue=len(pairs_final_test))
    test_record.append(answer(test_record[0], max_len_a, max_len_q, wordtoix, start_token, end_token, GRU_units, encoder, decoder, ixtoword))
    pairs_final_test_new.append(test_record)

#Calculate the Loss mean of tested data in this epoche
test_loss = statistics.mean([loss_function(loss[1], loss[2]) for loss in pairs_final_test_new])
print(f'Test-Loss of epoche {tested_epoch}: {test_loss}')

#Save the tested results
history['lossTest'].append(test_loss)

with open('training_history.pkl', 'wb') as f:
    pickle.dump(history, f)
