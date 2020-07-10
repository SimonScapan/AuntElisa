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

#Removes special characters from text
def clean_text(text):
    '''Clean text by removing unnecessary characters and altering the format of words.'''

    text = text.lower()
    
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "that is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"how's", "how is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"n'", "ng", text)
    text = re.sub(r"'bout", "about", text)
    text = re.sub(r"'til", "until", text)
    text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text)
    
    return text

#Creates a progress bar
def progressBar(value, endvalue, bar_length=20, job='Job'):

    percent = float(value) / endvalue
    arrow = '-' * int(round(percent * bar_length)-1) + '>'
    spaces = ' ' * (bar_length - len(arrow))

    sys.stdout.write("\r{0} Completion: [{1}] {2}%".format(job,arrow + spaces, int(round(percent * 100))))
    sys.stdout.flush()

#Splits the data into train (80%), test (5%) and valid(15%)
def split_dataset(sentences, sizes = [0.8, 0.5, 0.15] ): # number of examples
    data_len = len(sentences)
    lens = [int(data_len*item) for item in sizes]

    train = sentences[:lens[0]]
    test = sentences[lens[0]:lens[0]+lens[1]]
    test_new = []
    for sentences in test:
      test_new.append([re.sub('[<>BEOS]', '', sentences[0])[1:-1], re.sub('[<>BEOS]', '', sentences[1])[1:-1]])
    valid = sentences[-lens[-1]:]

    return train, test_new, valid

#Making the embedding mtrix and decide whether to use pretrained word embeddings
def make_embedding_layer(vocab_len, wordtoix, embedding_dim=100, glove=True):
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

#Similar to the training loop, but without teacher forcing here. Input to the decoder at each time step is its previous predictions with the hidden state and encoder output
def evaluate(sentence, max_len_a, max_len_q, wordtoix, start_token, end_token, GRU_units, encoder, decoder, ixtoword):
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
def answer(sentence, max_len_a, max_len_q, wordtoix, start_token, end_token, GRU_units, encoder, decoder, ixtoword, training=False):
    try:
      result, sentence, attention_plot = evaluate(sentence, max_len_a, max_len_q, wordtoix, start_token, end_token, GRU_units, encoder, decoder, ixtoword)
      return result
    except KeyError:
      return "Sorry, I did not understand you. Please say something else"
#Define the loss function
def loss_function(real, pred):

    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = K.sparse_categorical_crossentropy(real, pred, from_logits= False)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)

#Define the tensorflow trining function
@tf.function
def train_step(inp, targ, enc_hidden, encoder, wordtoix, start_token, batch_size, decoder, optimizer):
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

#Define a test which is printed after each train epoche
def test_bot(max_len_a, max_len_q, wordtoix, start_token, end_token, GRU_units, encoder, decoder, ixtoword):
    print('#'*20)
    q = 'Hello'
    print('Greedy| Q:',q,'?  A:',answer(q, max_len_a, max_len_q, wordtoix, start_token, end_token, GRU_units, encoder, decoder, ixtoword, training=True))
    print('%')
    q = 'How are you'
    print('Greedy| Q:',q,'?  A:',answer(q, max_len_a, max_len_q, wordtoix, start_token, end_token, GRU_units, encoder, decoder, ixtoword, training=True))
    print('%')
    q= 'Are you my friend'
    print('Greedy| Q:',q,'?  A:',answer(q, max_len_a, max_len_q, wordtoix, start_token, end_token, GRU_units, encoder, decoder, ixtoword, training=True))
    print('%')
    q = 'What are you doing'
    print('Greedy| Q:',q,'?  A:',answer(q, max_len_a, max_len_q, wordtoix, start_token, end_token, GRU_units, encoder, decoder, ixtoword, training=True))
    print('%')
    q = 'What is your favorite restaurant'
    print('Greedy| Q:',q,'?  A:',answer(q, max_len_a, max_len_q, wordtoix, start_token, end_token, GRU_units, encoder, decoder, ixtoword, training=True))
    print('%')
    q = 'Who are you'
    print('Greedy| Q:',q,'?  A:',answer(q, max_len_a, max_len_q, wordtoix, start_token, end_token, GRU_units, encoder, decoder, ixtoword, training=True))
    print('%')
    q = 'Do you want to go out'
    print('Greedy| Q:',q,'?  A:',answer(q, max_len_a, max_len_q, wordtoix, start_token, end_token, GRU_units, encoder, decoder, ixtoword, training=True))
    print('#'*20)

#Create plot to show the loss progress while training
def plot_history(best_ep, smallest_loss, history):
    plt.figure(figsize=(4,3))
    plt.plot(best_ep,smallest_loss,'ro')
    plt.plot(history['loss'],'b-')
    plt.plot(history['lossTest'],'b--')
    plt.legend(['best','loss'])
    plt.show()