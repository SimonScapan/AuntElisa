# -*- coding: utf-8 -*-
import re
#Defining functions which may be needed in multiple files

#Splits the data into train (70%), test (15%) and valid(15%)
def split_dataset(input, output, sizes = [0.7, 0.15, 0.15] ): # number of examples
    data_len = len(input)
    lens = [int(data_len*item) for item in sizes]

    trainX, trainY = input[:lens[0]], output[:lens[0]]
    testX, testY = input[lens[0]:lens[0]+lens[1]], output[lens[0]:lens[0]+lens[1]]
    validX, validY = input[-lens[-1]:], output[-lens[-1]:]

    return (trainX,trainY), (testX,testY), (validX,validY)

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