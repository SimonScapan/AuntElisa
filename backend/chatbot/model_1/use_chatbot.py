# -*- coding: utf-8 -*-
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.models.seq2seq import Seq2seq
import pickle
import random
from functions import clean_text

with open('metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)

word2idx = metadata['wordindex']   # dict  word 2 index
idx2word = metadata['index2word']   # list index 2 word

unk_id = idx2word.index('unk')   # 1
pad_id = idx2word.index('_')

vocabulary_size = len(idx2word)
end_id = vocabulary_size + 1

#Needs to be customized to the model in training file
model_ = Seq2seq(
        decoder_seq_length = 20,
        cell_enc=tf.keras.layers.GRUCell,
        cell_dec=tf.keras.layers.GRUCell,
        n_layer=3,
        n_units=256,
        embedding_layer=tl.layers.Embedding(vocabulary_size=vocabulary_size + 2, embedding_size=1024),
        )

load_weights = tl.files.load_npz(name='model_epoche29.npz')
tl.files.assign_weights(load_weights, model_)

#preprocessing of input: manipulates the text to lower case and only keeps the whitelist       
def preprocessing(input_string):
    return clean_text(input_string.lower())

#Prints on of the most probable answer
def AuntEliza(input, top_n): 
    model_.eval()
    input_id = [word2idx.get(w, unk_id) for w in preprocessing(input).split(" ")]
    sentence_id = model_(inputs=[[input_id]], seq_length=20, start_token=vocabulary_size, top_n = top_n)
    sentence = []
    for w_id in sentence_id[0]:
        if int(w_id) == end_id:
            break
        w = idx2word[w_id]
        sentence = sentence + [w]
    print('AuntEliza: ' + ' '.join(sentence))

message = ''

while message != 'stop':
    message = input('You: ')
    AuntEliza(message, 5)