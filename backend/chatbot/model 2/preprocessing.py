import csv
import re
import pickle
import random
from functions import clean_text

#Define Parameters of Preprocessing
maxlength = 20
minlength = 1
number_sample_conversations = 6000

#Load conversations file and get list with lists of L-ids
conversations_file = open("../data/conversations.txt")
read = list(csv.reader(conversations_file))
dialogs = [re.findall(r'L.\d*',str(i)) for i in read]    
        
#Loade lines
lines_file = open("../data/lines.txt", encoding="utf-8")
lines = list(csv.reader(lines_file, delimiter ='\n'))

#Create a list that contains lists of length 2: each first element is a L-id and second element is corresponding text
content = []
for i in lines:
    line = re.findall('(L.\d*).*\+\+\+\$\+\+\+.*\+\+\+\$\+\+\+ (.*)', i[0])
    try:
        content.append([line[0][0], line[0][1]])
    except IndexError:
            pass

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

#Cleans the text, creates a List of too long sentences and creates a List of tokenized sentences which are not too long
tooLong=[]
tokenized_sentences = []
contentdict = {}
for line in content:
    words = clean_text(line[1]).split(' ')
    if '' in words:
        words.remove('')
    if len(words) > maxlength:
        tooLong.append(line[0])
    else:
        tokenized_sentences.append(words)
        contentdict[line[0]] = ' '.join(words)

#create set of words
wordset = []
for sentence in tokenized_sentences:
    for word in sentence:
        wordset.append(word)
wordset = list(set(wordset))

#creates an unique index for each word in two dictionarys: First uses words as keys and index as value and the second one switches them
word2index = {}
index2word = {}
counter = 1

for word in wordset:
    word2index[word] = counter
    index2word[counter] = word
    counter += 1

word2index['<BOS> '] = counter+1
index2word[counter+1] = '<BOS> '
word2index[' <EOS>'] = counter+2
index2word[counter+2] = ' <EOS>'

wordset = wordset + ['<BOS> '] + [' <EOS>']

#Creating a List of a List with messages represented by a List of word IDs and the corresponding response represented by a list of word IDs
messageResponse = []
for dialog in dialogs:
    for i in range(len(dialog)):
        try:
            if dialog[i] not in tooLong and dialog[i+1] not in tooLong: 
                messageResponse.append(['<BOS> ' + contentdict[dialog[i]] + ' <EOS>', '<BOS> ' + contentdict[dialog[i+1]] + ' <EOS>'])
        except IndexError:
            pass

preprocessed_data= {'word2ix' : word2index, 'ixtoword' : index2word, 'pairs_final_train': random.shuffle(messageResponse), 'short_vocab': wordset, 'max_len_q': maxlength}

#Writes the preprocessed data to file
with open('preprocessed_data.pkl', 'wb') as f:
        pickle.dump(preprocessed_data, f)
print("Saved preprocessed data successfully!")