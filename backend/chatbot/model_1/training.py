import tensorflow as tf
import tensorlayer as tl
import numpy as np
from tensorlayer.cost import cross_entropy_seq, cross_entropy_seq_with_mask
from tqdm import tqdm
from sklearn.utils import shuffle
from tensorlayer.models.seq2seq import Seq2seq
from tensorlayer.models.seq2seq_with_attention import Seq2seqLuongAttention
import os
import pickle
from functions import split_dataset

def load_data():
    # read data control dictionaries
    with open('metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    # read numpy arrays
    idx_q = np.load('idxm.npy')
    idx_a = np.load('idxr.npy')
    return metadata, idx_q, idx_a

metadata, idx_q, idx_a = load_data()
(trainX, trainY), (testX, testY), (validX, validY) = split_dataset(idx_q, idx_a)
trainX = tl.prepro.remove_pad_sequences(trainX.tolist())
trainY = tl.prepro.remove_pad_sequences(trainY.tolist())
testX = tl.prepro.remove_pad_sequences(testX.tolist())
testY = tl.prepro.remove_pad_sequences(testY.tolist())
validX = tl.prepro.remove_pad_sequences(validX.tolist())
validY = tl.prepro.remove_pad_sequences(validY.tolist())

# Parameters
src_len = len(trainX)
tgt_len = len(trainY)

#Define Hyperparameters
batch_size = 32 #Choose batch size
emb_dim = 1024 #Choss embedding size for word vectors
n_step = src_len // batch_size
src_vocab_size = len(metadata['index2word'])
last_stopped =  0 #0 if you are training the model from scratch
num_epochs = 50 - last_stopped #Number of training epoches
n_layer = 3 #Choose number of Layers
n_units=256 #Choose number of units

#Add start and end tokens in sentences for training
word2idx = metadata['wordindex']   # dict  word 2 index
idx2word = metadata['index2word']   # list index 2 word

unk_id = idx2word.index('unk')   # 1
pad_id = idx2word.index('_')     # 0

start_id = src_vocab_size  # 8002
end_id = src_vocab_size + 1  # 8003

word2idx.update({'start_id': start_id})
word2idx.update({'end_id': end_id})
idx2word = idx2word + ['start_id', 'end_id']

src_vocab_size = tgt_vocab_size = src_vocab_size + 2

vocabulary_size = src_vocab_size

#Define length of sentences
decoder_seq_length = 20

#Define model
model_ = Seq2seq(
    decoder_seq_length = decoder_seq_length,
    cell_enc=tf.keras.layers.GRUCell,
    cell_dec=tf.keras.layers.GRUCell,
    n_layer=n_layer,
    n_units=n_units,
    embedding_layer=tl.layers.Embedding(vocabulary_size=vocabulary_size, embedding_size=emb_dim),
    )

# Try to load model to continue training
try:
    load_weights = tl.files.load_npz(name=f'model_epoche{last_stopped}.npz')
    tl.files.assign_weights(load_weights, model_)
    print(f'Loaded Epoche {last_stopped} successfully!')
except:
    pass

#Define optimizer and learning rate
optimizer = tf.optimizers.Adam(learning_rate=0.001)
model_.train()

for epoch in range(num_epochs):
    model_.train()
    trainX, trainY = shuffle(trainX, trainY, random_state=0)
    total_loss, n_iter = 0, 0
    for X, Y in tqdm(tl.iterate.minibatches(inputs=trainX, targets=trainY, batch_size=batch_size, shuffle=False), 
                    total=n_step, desc='Epoch[{}/{}]'.format(epoch + 1, num_epochs), leave=False):

        #Recreate the padding and start and end tokens
        X = tl.prepro.pad_sequences(X)
        _target_seqs = tl.prepro.sequences_add_end_id(Y, end_id=end_id)
        _target_seqs = tl.prepro.pad_sequences(_target_seqs, maxlen=decoder_seq_length)
        _decode_seqs = tl.prepro.sequences_add_start_id(Y, start_id=start_id, remove_last=False)
        _decode_seqs = tl.prepro.pad_sequences(_decode_seqs, maxlen=decoder_seq_length)
        _target_mask = tl.prepro.sequences_get_mask(_target_seqs)

        with tf.GradientTape() as tape:
            #Compute outputs
            output = model_(inputs = [X, _decode_seqs])
            
            output = tf.reshape(output, [-1, vocabulary_size])
            #Compute loss and update model
            loss = cross_entropy_seq_with_mask(logits=output, target_seqs=_target_seqs, input_mask=_target_mask)

            grad = tape.gradient(loss, model_.all_weights)
            optimizer.apply_gradients(zip(grad, model_.all_weights))
    
        total_loss += loss
        n_iter += 1

    #Save the current model to load again if algorithem was interrupted
    tl.files.save_npz(model_.all_weights, name=f'model_epoche{epoch+1+last_stopped}.npz')
    os.remove(f'model_epoche{epoch+last_stopped}.npz', dir_fd=None) 

#Save the final model
os.remove(f'model_epoche{epoch+last_stopped}.npz', dir_fd=None) 
tl.files.save_npz(model_.all_weights, name='model_final.npz')