#Import packages
import numpy as np
import re
import tensorflow as tf
import time

#Import functions and classes
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, GRU, Embedding, Dropout, Bidirectional

#Define the encoder of the network
class Encoder(tf.keras.Model):
    #Define the parameters of the class
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_size, max_len_q, embeddings):
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

class Decoder(tf.keras.Model):
    #Define the parameters of the decoder class
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_size, embeddings):
        super(Decoder, self).__init__()
        self.batch_sz = batch_size
        self.embeddings = embeddings
        self.units = 2 * dec_units #As we are using an bidirectional encoder
        self.fc = Dense(vocab_size, activation='softmax', name='dense_layer')
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

