import csv
import re
import numpy as np
import pickle
from functions import clean_text

#Define the max and min length of a message or response in words(improves perfomance and faster training)
maxlength = 20
minlength = 1

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

#manipulates the text to lower cas and only keeps the whitelist. Saves it in a list that contains lists of length 2: each first element is L-id and second element is corresponding text        
counter=0    
for line in content:
    content[counter][1] = clean_text(line[1])
    counter += 1

#creates a List of too long texts and creates a List of tokenized sentences which are not too long
tooLong=[]
tokenized_sentences = []
tokenized_sentences_all = []

for line in content:
    words = line[1].split(' ')
    if '' in words:
        words.remove('')
    if len(words) > maxlength:
        tooLong.append(line[0])
    else:
        tokenized_sentences.append(words)
    tokenized_sentences_all.append(words)

#create set of words
wordset = []
for sentence in tokenized_sentences_all:
    for word in sentence:
        wordset.append(word)
wordset = list(set(wordset))

#creates an unique index for each word in two dictionarys: First uses words as keys and index as value and the second one switches them
wordindex = {}
indexword = {}
counter = 1

for word in wordset:
    wordindex[word] = counter
    indexword[counter] = word
    counter += 1
    
#creats a dict which uses L-Id of each sentence as key and uses the content of the sentence(words are now replaced by corrresponding ID) as value and filled with nulls to the max size (+one 0 additionally as stop token)
indexcontent = {}
for text in content:
    sentence = text[1].split(' ')
    if '' in text:
        sentence.remove('')
    indexsentence = []
    for word in sentence:
        indexsentence.append(wordindex[word])
    while len(indexsentence) < maxlength+1:
        indexsentence.append(0)
    indexcontent[text[0]] = indexsentence 

#creates a dictionary with message represted by IDs of words and the corresponding response represented by IDs of words
indexmessageResponse = []
for dialog in dialogs:
    for i in range(len(dialog)):
        try:
            if dialog[i] not in tooLong and dialog[i+1] not in tooLong: 
                indexmessageResponse.append([indexcontent[dialog[i]], indexcontent[dialog[i+1]]])
        except IndexError:
            pass

#creates a list which contains all messages with word Ids instead of words
#creates a list which contains all responses with word Ids instead of words
idxm = []
idxr = []
for conversation in indexmessageResponse:
    idxm.append(conversation[0])
    idxr.append(conversation[1])
    
#Creates a list of all unique words, a place holder for unknown words (in case of input od unkown words) and _
index2word = ['_'] + ['unk'] + [ x for x in wordindex.keys() ]

metadata= {'wordindex' : wordindex, 'index2word' : index2word}

#Writes the index lists of message and response and the metadata to file
np.save('idxm.npy', np.array(idxm, dtype=np.int32))
np.save('idxr.npy', np.array(idxr, dtype=np.int32))
with open('metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)

print('Saved List of message, rsponse and the metadata successfully.')