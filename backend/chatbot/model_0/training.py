import csv
import re
import numpy as np
import nltk
import pickle
import operator
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding
from keras.layers import Input, Dense, LSTM, TimeDistributed
from keras.models import Model
from keras.utils import plot_model
from keras.models import model_from_json
import matplotlib.pyplot as plt

#Define Parameters from Preprocessing
maxlength = 20
minlength = 1
number_sample_conversations = 6000

#Load the preprocessed data
def load_data():
    # read data control dictionaries
    with open('preprocessed_data.pkl', 'rb') as f:
        data = pickle.load(f)
    return data

data = load_data()

index2word = data['index2word']
word2index = data['word2index']
encoder_sequences = data['encoder_sequences']
decoder_sequences = data['decoder_sequences']

#Define Size of output data
VOCAB_SIZE = len(index2word) + 1
MAX_LEN = 15
num_samples = number_sample_conversations
decoder_output_data = np.zeros((num_samples, MAX_LEN, VOCAB_SIZE), dtype="float32")

#Create the input matrix
encoder_input_data = pad_sequences(encoder_sequences, maxlen=MAX_LEN, dtype='int32', padding='post', truncating='post')
decoder_input_data = pad_sequences(decoder_sequences, maxlen=MAX_LEN, dtype='int32', padding='post', truncating='post')

#Loading pretraines vordvectors to improve the training result
embeddings_index = {}
with open('../data/glove.6B.50d.txt', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

print("Glove Loded!")

#Create embedding matrix and embedding layer for word2vec
embedding_dimention = 50
def embedding_matrix_creater(embedding_dimention, word_index):
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dimention))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
          # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return embedding_matrix
  
embedding_matrix = embedding_matrix_creater(50, word_index=word2index)

embed_layer = Embedding(input_dim=VOCAB_SIZE, output_dim=50, trainable=True,)
embed_layer.build((None,))
embed_layer.set_weights([embedding_matrix])

#Define Number of hidden dimensions of the RNN
HIDDEN_DIM = 300

#Define and create the Layers of the Seq2seq model and print a summery of the Seq2Seq model
def seq2seq_model_builder(HIDDEN_DIM):
    
    encoder_inputs = Input(shape=(MAX_LEN, ), dtype='int32',)
    encoder_embedding = embed_layer(encoder_inputs)
    encoder_LSTM = LSTM(HIDDEN_DIM, return_state=True)
    encoder_outputs, state_h, state_c = encoder_LSTM(encoder_embedding)
    
    decoder_inputs = Input(shape=(MAX_LEN, ), dtype='int32',)
    decoder_embedding = embed_layer(decoder_inputs)
    decoder_LSTM = LSTM(HIDDEN_DIM, return_state=True, return_sequences=True)
    decoder_outputs, _, _ = decoder_LSTM(decoder_embedding, initial_state=[state_h, state_c])
    
    # dense_layer = Dense(VOCAB_SIZE, activation='softmax')
    outputs = TimeDistributed(Dense(VOCAB_SIZE, activation='softmax'))(decoder_outputs)
    model = Model([encoder_inputs, decoder_inputs], outputs)
    
    return model

outputs = TimeDistributed(Dense(VOCAB_SIZE, activation='softmax'))(decoder_outputs)     
model = Model([encoder_inputs, decoder_inputs], outputs)

model = seq2seq_model_builder(HIDDEN_DIM)
model.summary()

#Define Training of the Seq2Seq model
model.compile(optimizer='adam', loss ='categorical_crossentropy', metrics = ['accuracy'])

BATCH_SIZE = 60
EPOCHS = 10

print("Shape of the Training input data: " + str(encoder_input_data.shape))

#Train the Seq2Seq Model
history = model.fit([encoder_input_data, decoder_input_data], 
                     decoder_output_data, 
                     epochs=EPOCHS, 
                     batch_size=BATCH_SIZE)

#Print the training accuracy
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'])
#plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
plt.show()

#Print the training loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
plt.show()

#Save the Weights of the Seq2Seq Model
json_string = model.to_json()
open('seq2seq.json', 'w').write(json_string)
model.save_weights('seq2seq_weights.h5')

json_string = model.to_json()
open('seq2seq.json', 'w').write(json_string)
model.save_weights('seq2seq_weights.h5')

# Save the weights
model_json = model.to_json()
with open("seq2seq.json", "w") as json_file:
    json_file.write(model_json)


model.load_weights('seq2seq_weights.h5')
print("Saved Model!")

#Save the Model
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("chatbot_model.h5")
print("Saved Model!")

from keras.models import model_from_json
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
chat_model = model_from_json(loaded_model_json)

#Load weights into new model
chat_model.load_weights("chatbot_model.h5")