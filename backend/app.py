# coding=utf8
from flask import Flask, request
from flask_cors import CORS
import pickle
import tensorflow as tf
from chatbot.model_2.functions import answer, make_embedding_layer
from chatbot.model_2.classes import Decoder, Encoder

# define test environment
GRU_units = 256
batch_size = 32
emb_dim = 50

# load the preprocessed data
with open('chatbot/model_2/preprocessing/preprocessed_data.pkl', 'rb') as f:
        preprocessed_data = pickle.load(f)
wordtoix = preprocessed_data['word2ix']
ixtoword = preprocessed_data['ixtoword']
pairs_final = preprocessed_data['pairs_final_train']
short_vocab = preprocessed_data['short_vocab']
max_len_q = preprocessed_data['max_len_q']
max_len_a = preprocessed_data['max_len_q']

# define parameters to use the model
end_token = '<EOS>'
start_token = '<BOS>'
pad_token = 'pad0'
ixtoword[0] = pad_token
vocab_len = len(short_vocab) + 2
optimizer = tf.keras.optimizers.Adam(0.0005)
embeddings = make_embedding_layer(vocab_len=vocab_len, wordtoix=wordtoix, embedding_dim=emb_dim, glove=False)
encoder = Encoder(vocab_len, emb_dim, GRU_units, batch_size, max_len_q, embeddings)
decoder = Decoder(vocab_len, emb_dim, GRU_units, batch_size, embeddings)

# load the current tensorflow model
checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)
manager = tf.train.CheckpointManager(checkpoint, 'chatbot/model_2/training/model', max_to_keep = 300)
checkpoint.restore(manager.latest_checkpoint)

# initialize Flask
APP = Flask(__name__, static_folder="build/static", template_folder="build")
CORS = CORS(APP)

# listen to GET method and compute input text with chatbot and give back to frontend
@APP.route('/backend/<text>', methods=["GET"])
def chatbot(text):
    response = str(answer(str(text), max_len_a, max_len_q, wordtoix, start_token, end_token, GRU_units, encoder, decoder, ixtoword))
    return response

# run APP on localhost
if __name__ == '__main__':
    APP.run(debug=True, host='0.0.0.0')


