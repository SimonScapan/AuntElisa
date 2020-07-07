#Import packages
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os
import pandas as pd
import pickle
import random
import re
import string
import sys
import tensorflow as tf
import time

#Import functions and classes
from functions import clean_text, split_dataset
from nltk.translate.bleu_score import corpus_bleu
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import add, Input, Dense, LSTM, GRU, TimeDistributed, Embedding, Dropout, Bidirectional, Concatenate, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from training import Encoder, make_embedding_layer, BahdanauAttention, Decoder, evaluate, plot_attention, answer, loss_function, train_step, plot_history

#Load the preprocessed data
with open('/content/drive/My Drive/Data Exploration Project/preprocessed_data.pkl', 'rb') as f:
        preprocessed_data = pickle.load(f)

wordtoix = preprocessed_data['word2ix']
ixtoword = preprocessed_data['ixtoword']
pairs_final = preprocessed_data['pairs_final_train']
short_vocab = preprocessed_data['short_vocab']

max_len_q = preprocessed_data['max_len_q']
max_len_a = preprocessed_data['max_len_q']

end_token = ' <EOS>'
start_token = '<BOS> '

np.random.seed(1)
random.seed(1)

snapshot_folder = '/content/drive/My Drive/Data Exploration Project'

#Define some Hyperparameters
test = False
print('test: ', test)
if test:
    batch_size = 4
    emb_dim = 10
else:
    batch_size = 32
    emb_dim = 50

init_lr = 0.0005

#Load predefined hyperparameter compinations
hyperparameters = pd.read_csv('/content/drive/My Drive/Data Exploration Project/hyperparameterValidation.csv' , delimiter=';')

#Iterate over the diffrent predefined hyperparameter combinations to validate
for combination in len(hyperparameters):
  batch_per_epoche = int(hyperparameters.loc[combination][0])
  GRU_units = int(hyperparameters.loc[combination][1])

  #Since index 0 is used as padding, we have to increase the vocab size
  vocab_len = len(short_vocab) + 2

  #Splits the data into train (70%), test (15%) and valid(15%)
  pairs_final_train, pairs_final_test, pairs_final_valid = split_dataset(pairs_final)

  #Making the embedding matrix and decide whether to use pretrained word embeddings
  embeddings = make_embedding_layer(embedding_dim=emb_dim, glove=not test)

  #Create encoder
  encoder = Encoder(vocab_len, emb_dim, GRU_units)

  decoder = Decoder(vocab_len, emb_dim, GRU_units)

  #Applying the adam optimizer to update the network weights
  optimizer = tf.keras.optimizers.Adam(init_lr)

  #Initalize a checkpoint to save the model later
  checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)

  #Define and calculate hyperparameters for training
  history={'loss':[]}
  smallest_loss = np.inf
  best_ep = 1
  EPOCHS = 7 # but 150 is enough
  enc_hidden = encoder.initialize_hidden_state()
  steps_per_epoch = len(pairs_final_valid)//batch_size # used for caculating number of batches
  current_ep = 1

  #Reload a checkpoint if training was interrupted
  last_stopped = 0
  try:
    checkpoint.restore(snapshot_folder+'/'+str(emb_dim)+"-ckpt-"+str(last_stopped))
    print('Successfully loaded epoche' + str(last_stopped))
  except:
    pass

  batch_loss = K.constant(0)
  X, y = [], []


  #Iterate over the training epoches
  for ep in range(current_ep,EPOCHS-last_stopped):
      current_ep = ep    
      start = time.time()
      total_loss = 0
      btch = 1

      #Iterates over each pair of conversation
      for p in pairs_final_valid:     
          
          #Split the conversation into message and response
          question = p[0]
          label = p[1]

          #Create lists of indices out of the sentences
          question_seq = [wordtoix[word] for word in question.split(' ') if word in wordtoix]
          label_seq = [wordtoix[word] for word in label.split(' ') if word in wordtoix]

          #Padding of the sentences to the max length
          enc_in_seq = pad_sequences([question_seq], maxlen=max_len_q, padding='post')[0]
          dec_out_seq = pad_sequences([label_seq], maxlen=max_len_a, padding='post')[0]
          
          X.append(enc_in_seq)
          y.append(dec_out_seq)


          if len(X) == batch_size:
              #Put the batch sized input and output arrays and the enc_hidden into the training step
              batch_loss = train_step(np.array(X), np.array(y), enc_hidden)

              #Sum up to the total loss in each step
              total_loss += batch_loss

              #Clear X and y to be able to create the next batch
              X , y = [], []
              btch += 1

              #After calculating to given number of batches per epoche, print metadata and loss
              if btch % (steps_per_epoch//batch_per_epoche) == 0:
                  print('Epoch {} Batch {} Loss: {:.4f}'.format(ep + last_stopped , btch, K.get_value(batch_loss)))

      #Calculate average loss of each epoches step
      epoch_loss =  K.get_value(total_loss) / steps_per_epoch
      print('\n*** Epoch {} Loss {:.4f} ***\n'.format(ep + last_stopped,epoch_loss))
      history['loss'].append(epoch_loss)

      #Track each epoches average loss to find the best epoche while training
      if epoch_loss < smallest_loss:
          smallest_loss = epoch_loss
          best_ep = ep 
          print('Reached a new best loss!')
      
      #plot after each third epoche
      if ep % 3 == 0:
          plot_history()
          
      print('Best epoch so far: ',best_ep,' smallest loss:',smallest_loss)
      print('Time taken for the epoch {:.3f} sec\n'.format(time.time() - start))

      print('=' * 40)

  #Read the hyperparameter file again and save each loss  
  new_hyperparameters = pd.read_csv('/content/drive/My Drive/Data Exploration Project/hyperparameterValidation.csv' , delimiter=';')
  new_hyperparameters['loss'][combination] = epoch_loss
  new_hyperparameters.to_csv('/content/drive/My Drive/Data Exploration Project/hyperparameterValidation.csv' ,sep = ';', index=False)