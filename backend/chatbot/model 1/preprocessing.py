import csv
import re
import numpy as np
from tqdm import tqdm
import nltk
import itertools
import pickle

#Define the max and min length of a message or response in words(improves perfomance and faster training)
maxlength = 20
minlength = 1

#Load conversations file and get list with lists of L-ids
conversations_file = open("conversations.txt")
read = list(csv.reader(conversations_file))
dialogs = [re.findall(r'L.\d*',str(i)) for i in read]    

'''#Creates a dict that contains each messages L-ID as key and the corresponding response as L-ID
messageResponse = {}
for dialog in dialogs:
    for i in range(len(dialog)):
        try:
            messageResponse[dialog[i]]= dialog[i+1]
        except IndexError:
            pass'''
        
#Loade lines
lines_file = open("lines.txt", encoding="utf-8")
lines = list(csv.reader(lines_file, delimiter ='\n'))

#Create a list that contains lists of length 2: each first element is a L-id and second element is corresponding text
content = []
for i in lines:
    line = re.findall('(L.\d*).*\+\+\+\$\+\+\+.*\+\+\+\$\+\+\+ (.*)', i[0])
    try:
        content.append([line[0][0], line[0][1]])
    except IndexError:
            pass

#Define Characters to keep
whitelist = '0123456789abcdefghijklmnopqrstuvwxyz ' # space is included in whitelist

#manipulates the text to lower cas and only keeps the whitelist. Saves it in a list that contains lists of length 2: each first element is L-id and second element is corresponding text        
counter=0    
for line in content:
    newLine = ''
    for character in line[1].lower():
        if character in whitelist:
            newLine = newLine + character
    content[counter][1] = newLine
    counter += 1

'''#Creates a List that contains lists of length two: first element is the message as L-ID and second element is the corresponding response as L-ID
messageResponseList = []
for dialog in dialogs:
    for i in range(len(dialog)):
        try:
            messageResponseList.append([dialog[i], dialog[i+1]])
        except IndexError:
            pass'''

print(62)

#creates a List of too long texts and creates a List of tokenized sentences which are not too long
tooLong=[]
tokenized_sentences = []
tokenized_sentences_all = []
print('creating a List of too long texts and create a List of tokenized sentences which are not too long')
for line in tqdm(content):
    words = line[1].split(' ')
    if '' in words:
        words.remove('')
    if len(words) > maxlength:
        tooLong.append(line[0])
    else:
        tokenized_sentences.append(words)
    tokenized_sentences_all.append(words)

'''#Removes too long messages and responses from the message response list
print('Removing too long messages and responses')
for line in tqdm(tooLong):
    for x in messageResponseList:
        if line in x:
            messageResponseList.remove(x)'''

'''#counts the occurences of words in a dictionary (includes also words from too long sentences)
wordCount={}
for line in content:
    words = line[1].split(' ')
    if '' in words:
        words.remove('')
    for word in words:
        if word in wordCount.keys():
            wordCount[word] += 1
        else:
            wordCount[word] = 1

#creates a list out of the worrdcount dictionary
countWord=[]
for key, value in wordCount.items():
    countList = [key,value]
    countWord.append(countList)'''

print(100)

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

print(124)

#creates a dictionary with message represted by IDs of words and the corresponding response represented by IDs of words
indexmessageResponse = []
print('creating a dictionary with message represted by IDs of words and the corresponding response represented by IDs of words')
for dialog in tqdm(dialogs):
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

#Saves the min and max limits 
limit = {'maxq' : maxlength, 'minq' : minlength, 'maxa' : maxlength, 'mina' : minlength}

print(156)

'''#creates a List of too long texts and creates a List of tokenized sentences which are not too long
tokenized_sentences = []
for line in content:
    words = line[1].split(' ')
    if '' in words:
        words.remove('')
    if len(words) > maxlength:
        tooLong.append(line[0])
    else:
        tokenized_sentences.append(words)'''

#Creates a frequency distribution of each unique word        
freq_dist = nltk.FreqDist(itertools.chain(*tokenized_sentences))


metadata= {'wordindex' : wordindex, 'index2word' : index2word, 'limit' : limit, 'freq_dist' : freq_dist}

print(165)
#Writes the index lists of message and response and the metadata to file
np.save('idxm.npy', np.array(idxm, dtype=np.int32))
np.save('idxr.npy', np.array(idxr, dtype=np.int32))
with open('metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)
print('Saved List of message, rsponse and the metadata successfully.')