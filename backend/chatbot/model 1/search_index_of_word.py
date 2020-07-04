import pickle

#Open metadate which also contains the mapping from word to its id
with open('metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)

#Input one word to search if it is represented in dataset
word = input('Proofe word:')

#Try to access the word in mapping and print its id if it exists
try:
    metadata['wordindex'][word]
    print('Word exists! Index: ' + str(metadata['wordindex'][word]))
except KeyError:
    print("Word doesn't exist!")