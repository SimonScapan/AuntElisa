import csv
import numpy as np
import re

def readMovieData():
    #read the raw data which contains the content of the conversaitions
    #manually deleted " and html commands like <b> from lines.txt
    lines_file = open("lines.txt", encoding="utf-8")
    lines = list(csv.reader(lines_file, delimiter ='\n'))

    #read the raw data which contains the structure of the conversaitions 
    conversations_file = open("conversations.txt")
    read = list(csv.reader(conversations_file))
    dialogs = [re.findall(r'L.\d*',str(i)) for i in read]     

    #Dictionary which maps every single sentance to it's conversation ID    
    mapper={}
    for i in lines:
        content = re.findall('(L.\d*).*\+\+\+\$\+\+\+.*\+\+\+\$\+\+\+(.*)', i[0])
        mapper[content[0][0]] = content[0][1]

    #Creates a dictionary which contains a mapping between each message and its respons
    for dialog in dialogs:
        for i in range(len(dialog)):
            try:
                conversationDictionary[mapper[dialog[i]]] = mapper[dialog[i+1]]
            except IndexError:
                pass


conversationDictionary = {}
readMovieData()

#Save the data in a numpy array which we will use later train and test the model
np.save('conversationDictionary.npy', conversationDictionary)

#Save the data in a big txt file
conversationFile = open('conversationData.txt', 'w')
for key, value in conversationDictionary.items():
    if (not key.strip() or not value.strip()):
        # If there are empty strings
        continue
    conversationFile.write(key.strip() + value.strip())