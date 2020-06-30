# -*- coding: utf-8 -*-
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.models.seq2seq import Seq2seq
import pickle
import random

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

load_weights = tl.files.load_npz(name='model.npz')
tl.files.assign_weights(load_weights, model_)

#Define Characters to keep
whitelist = '0123456789abcdefghijklmnopqrstuvwxyz ' # space is included in whitelist

#preprocessing of input: manipulates the text to lower case and only keeps the whitelist       
def preprocessing(input_string):
    counter=0    
    new_input = ''
    for character in input_string.lower():
        if character in whitelist:
            new_input = new_input + character
    return new_input

#Prints on of the n most probable answers
def AuntEliza(input, top_n): 
    model_.eval()
    input_id = [word2idx.get(w, unk_id) for w in preprocessing(input).split(" ")]
    answers = []
    for i in range(top_n):
        sentence_id = model_(inputs=[[input_id]], seq_length=20, start_token=vocabulary_size, top_n = top_n)
        sentence = []
        for w_id in sentence_id[0]:
            if int(w_id) == end_id:
                break
            w = idx2word[w_id]
            sentence = sentence + [w]
        answers.append(' '.join(sentence)) #List of 5 possible answers
    print('AuntEliza: ' + answers[random.randint(0, top_n)])

message = ''

while message != 'stop':
    message = input('You: ')
    AuntEliza(message, 5)