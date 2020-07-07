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

#Import functions
from functions import clean_text, split_dataset

np.random.seed(1)
random.seed(1)

snapshot_folder = '/content/drive/My Drive/Data Exploration Project'

#Define some hyperparameters
test = False
print('test: ', test)
if test:
    GRU_units = 10
    batch_size = 4
    emb_dim = 10
else:
    GRU_units = 20
    batch_size = 32
    emb_dim = 50

init_lr = 0.0005

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

#Splits the data into train (70%), test (15%) and valid(15%)
pairs_final_train, pairs_final_test, pairs_final_valid = split_dataset(pairs_final)

#Since index 0 is used as padding, we have to increase the vocab size
vocab_len = len(short_vocab) + 2

#Making the embedding mtrix and decide whether to use pretrained word embeddings
def make_embedding_layer(embedding_dim=100, glove=True):
    if glove == False:
        print('Just a zero matrix loaded')
        embedding_matrix = np.zeros((vocab_len, embedding_dim)) # just a zero matrix 
    else:
        print('Loading glove...')
        embeddings_index = {} 
        f = open('/content/drive/My Drive/Data Exploration Project/glove.6B.50d.txt', encoding="utf-8")
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()
        print("GloVe ",embedding_dim, ' loded!')
        #Get 200 dimensions dense vector for each of the vocab_rocc
        embedding_matrix = np.zeros((vocab_len, embedding_dim)) #To import as weights for Keras Embedding layer
        for word, i in wordtoix.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                #Words not found in the embedding index will be all zeros
                embedding_matrix[i] = embedding_vector
            
    embedding_layer = Embedding(vocab_len, embedding_dim, mask_zero=True, trainable=False) #We have a limited vocab so we 
                                                                                           #Do not train the embedding layer
                                                                                           #We use 0 as padding so mask_zero as True
    embedding_layer.build((None,))
    embedding_layer.set_weights([embedding_matrix])
    
    return embedding_layer

embeddings = make_embedding_layer(embedding_dim=emb_dim, glove=not test)

#Define the encoder of the network
class Encoder(tf.keras.Model):
    #Define the parameters of the class
    def __init__(self, vocab_size, embedding_dim, enc_units):
        super(Encoder, self).__init__()
        self.batch_sz = batch_size
        self.enc_units = enc_units
        self.embeddings = embeddings
        self.Bidirectional1 = Bidirectional(GRU(enc_units, return_sequences=True,
                               return_state=False, recurrent_initializer='glorot_uniform', name='gru_1'), name='bidirectional_encoder1')
        self.Bidirectional2 = Bidirectional(GRU(enc_units, return_sequences=True, 
                               return_state=True, recurrent_initializer='glorot_uniform', name='gru_2'), name='bidirectional_encoder2')                                                                                        
        self.dropout = Dropout(0.2)
        self.Inp = Input(shape=(max_len_q,)) # size of questions

    #Create bidirectional LSTM for easier access of information for previous and following layers 
    def bidirectional(self, bidir, layer, inp, hidden):
        return bidir(layer(inp, initial_state = hidden))
    
    #Create Embedding with dropout to reduce overfitting and increase model performance
    def call(self, x, hidden):
        x = self.embeddings(x)
        x = self.dropout(x)
        output, state_f,state_b = self.Bidirectional2(x)

        return output, state_f, state_b

    #Create an empty (or zero) tensor 
    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))

#Create encoder
encoder = Encoder(vocab_len, emb_dim, GRU_units)

#Create bahdanau attention in the network to find the correlation between massage and response
class BahdanauAttention(tf.keras.layers.Layer):
    #Define the parameters of the attention class
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)
        self.units = units
        
    def call(self, query, values):
        #Broadcast addition along the time axis to calculate the score
        query_with_time_axis = tf.expand_dims(query, 1)

        score = self.V(tf.nn.tanh(
            self.W1(query_with_time_axis) + self.W2(values)))
        
        #Create attention weights
        attention_weights = tf.nn.softmax(score, axis=1)
        #Apply attention weights
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights

#Define the decoder of the network
class Decoder(tf.keras.Model):
    #Define the parameters of the decoder class
    def __init__(self, vocab_size, embedding_dim, dec_units):
        super(Decoder, self).__init__()
        self.batch_sz = batch_size
        self.embeddings = embeddings
        self.units = 2 * dec_units #As we are using an bidirectional encoder
        self.fc = Dense(vocab_len, activation='softmax', name='dense_layer')
        #Use attention to improve creating responses
        self.attention = BahdanauAttention(self.units)
        self.decoder_gru_l1 = GRU(self.units, return_sequences=True, 
                                  return_state= False, recurrent_initializer='glorot_uniform' ,name='decoder_gru1')
        self.decoder_gru_l2 = GRU(self.units, return_sequences=False, 
                                  return_state= True, recurrent_initializer='glorot_uniform' ,name='decoder_gru2') 
        self.dropout = Dropout(0.4)
        
    def call(self, x, hidden, enc_output):
        #Get the attention weights
        context_vector, attention_weights = self.attention(hidden, enc_output)

        x = self.embeddings(x)

        #Concat input and context vector together
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1) 

        #Passing the concatenated vector to the GRU
        x = self.decoder_gru_l1(x)
        x = self.dropout(x)
        output, state = self.decoder_gru_l2(x)
        x = self.fc(output)
        return x, state, attention_weights

decoder = Decoder(vocab_len, emb_dim, GRU_units)

#Similar to the training loop, but without teacher forcing here. Input to the decoder at each time step is its previous predictions with the hidden state and encoder output
def evaluate(sentence):
    #Create a empty (or zero) matrix
    attention_plot = np.zeros((max_len_a, max_len_q))

    #Clean the input message
    sentence = clean_text(sentence)

    #Translate the message to its word indices with start and end token and fill with padding tokens
    inputs = [wordtoix[i] for i in sentence.split(' ')]
    inputs = [wordtoix[start_token]]+inputs+[wordtoix[end_token]]
    inputs = pad_sequences([inputs],maxlen=max_len_q, padding='post')

    inputs = tf.convert_to_tensor(inputs)

    result = ''

    #Create an empty hidden GRU layer (uses sigmoid activation)
    hidden = [tf.zeros((1, GRU_units))]

    #Apply the encoder on the input message 
    enc_out, enc_hidden_f, enc_hidden_b = encoder(inputs, hidden)

    #Conncatenate enc_hidden_f and enc_hidden_b
    dec_hidden = Concatenate(axis=-1)([enc_hidden_f, enc_hidden_b])

    #Add a new dimension for the index of the start token
    dec_input = tf.expand_dims([wordtoix[start_token]], 1)

    for t in range(max_len_a):
        #Use the decoder to create predictions
        predictions, dec_hidden, attention_weights = decoder(dec_input,
                                                             dec_hidden,
                                                             enc_out)

        #Reshape the attention weights
        attention_weights = tf.reshape(attention_weights, (-1, ))

        #Get all attention weights for the word in the current loop
        attention_plot[t] = K.get_value(attention_weights)
        
        #Get most probable next word based on the word in the current loop and previous predicted words
        predicted_id =  K.get_value(tf.argmax(predictions[0]))       

        #Stop prediction, if we reached the end token
        if ixtoword[predicted_id] == end_token:
            return result, sentence, attention_plot
        
        #Add the predicted word to the response
        result += ixtoword[predicted_id] + ' '

        #Put the predicted word back to the model as we are using RNN
        dec_input = tf.expand_dims([predicted_id], 1)

    return result, sentence, attention_plot

#Function for plotting the attention weights
def plot_attention(attention, sentence, predicted_sentence):
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention, cmap='viridis')

    fontdict = {'fontsize': 14}

    ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
    ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    
    plt.show()

#Retrieve the response based on the message
def answer(sentence, training=False):
    result, sentence, attention_plot = evaluate(sentence)

    if training:
        return result
    
    print('Input: %s' % (sentence))
    print('Predicted answer: {}'.format(result))
    attention_plot = attention_plot[1:len(result.split(' ')), :len(sentence.split(' '))]
    plot_attention(attention_plot, sentence.split(' '), result.split(' ')[:-1])

#Applying the adam optimizer to update the network weights
optimizer = tf.keras.optimizers.Adam(init_lr)

#Define the loss function
def loss_function(real, pred):

    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = K.sparse_categorical_crossentropy(real, pred, from_logits= False)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)

#Initalize a checkpoint to save the model later
checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)

#Define the tensorflow trining function
@tf.function
def train_step(inp, targ, enc_hidden):
    loss = 0

    with tf.GradientTape() as tape:

        #Call the Encoder class
        enc_output, enc_hidden_f, enc_hidden_b = encoder(inp, enc_hidden)
        
        #Conncatenate enc_hidden_f and enc_hidden_b
        dec_hidden = Concatenate(axis=-1)([enc_hidden_f, enc_hidden_b])

        #Add a new dimension for the index of the start token
        dec_input = tf.expand_dims([wordtoix[start_token]] * batch_size, 1)

        #Teacher forcing (feeding the target as the next input)
        for t in range(1, targ.shape[1]):
            
            #Passing enc_output to the decoder
            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
            #Each time just use one timestep output
            loss += loss_function(targ[:, t], predictions) 
            #Expected output at this time becomes input for next timestep
            dec_input = tf.expand_dims(targ[:, t], 1) 
            
    #Calculate the loss of this batch
    batch_loss = (loss / int(targ.shape[1]))

    #Calculate the gradients in respect to the variables
    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)

    #Apply the gradient to the optimizer to update the weights
    optimizer.apply_gradients(zip(gradients, variables))
    
    return batch_loss

#Define and calculate hyperparameters for training/testing
history={'loss':[]}
historyTest={'lossTest':[]}
smallest_loss = np.inf
best_ep = 1
EPOCHS = 5 # but 150 is enough
enc_hidden = encoder.initialize_hidden_state()
steps_per_epoch = len(pairs_final_train)//batch_size # used for caculating number of batches
current_ep = 1
batch_per_epoche = 6

#Define a test which is printed after each train epoche
def test_bot():
    print('#'*20)
    q = 'Hello'
    print('Greedy| Q:',q,'?  A:',answer(q, training=True))
    print('%')
    q = 'How are you'
    print('Greedy| Q:',q,'?  A:',answer(q, training=True))
    print('%')
    q= 'Are you my friend'
    print('Greedy| Q:',q,'?  A:',answer(q, training=True))
    print('%')
    q = 'What are you doing'
    print('Greedy| Q:',q,'?  A:',answer(q, training=True))
    print('%')
    q = 'What your favorite restaurant'
    print('Greedy| Q:',q,'?  A:',answer(q, training=True))
    print('%')
    q = 'Who are you'
    print('Greedy| Q:',q,'?  A:',answer(q, training=True))
    print('%')
    q = 'Do you want to go out'
    print('Greedy| Q:',q,'?  A:',answer(q, training=True))
    print('#'*20)

#Create plot to show the loss progress while training
def plot_history():
    plt.figure(figsize=(4,3))
    plt.plot(best_ep,smallest_loss,'ro')
    plt.plot(history['loss'],'b-')
    plt.plot(historyTest['lossTest'],'b--')
    plt.legend(['best','loss'])
    plt.show()

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
  for p in pairs_final_train:     
      
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
  
  #Test every 10th epoche based on test data
  if ep %10==0:
    pairs_final_test_new = []
    training = False
    test = True
    for test_record in pairs_final_test:
      test_record.append(answer(test_record[0]))
      pairs_final_test_new.append(test_record)
    test_loss = statistics.mean([loss_function(loss[1], loss[2]) for loss in final_test_new])
    print(test_loss)
    historyTest['lossTest'].append(test_loss)
    training = True
    test = False

  #Save the model of the current epoche
  checkpoint.save(file_prefix = '/content/drive/My Drive/Data Exploration Project')

  #Show how the bot is performing right now
  test_bot()

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

plot_history()
