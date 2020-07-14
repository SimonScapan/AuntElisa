import csv
import re
import pickle

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

#manipulates the text to lower case and only keeps the whitelist. Saves it in a list that contains lists of length 2: each first element is L-id and second element is corresponding text        
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
word2index = {}
index2word = {}
counter = 1

for word in wordset:
    word2index[word] = counter
    index2word[counter] = word
    counter += 1

word2index['<BOS>'] = counter+1
index2word[counter+1] = '<BOS>'
word2index['<EOS>'] = counter+2
index2word[counter+2] = '<EOS>'
word2index['<POS>'] = 0
index2word[0] = '<POS>'


#creats a dict which uses L-Id of each sentence as key and uses the content of the sentence(words are now replaced by corrresponding ID) as value and filled with nulls to the max size (+one 0 additionally as stop token)
indexcontent = {}
for text in content:
    sentence = text[1].split(' ')
    if '' in text:
        sentence.remove('')
    indexsentence = []
    for word in sentence:
        indexsentence.append(word2index[word])
    while len(indexsentence) < maxlength:
        indexsentence.append(0)
    indexcontent[text[0]] = indexsentence 

#Creating a List of a List with messages represented by a List of word IDs and the corresponding response represented by a list of word IDs
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
encoder_sequences = []
decoder_sequences = []
for conversation in indexmessageResponse:
    encoder_sequences.append(conversation[0])
    decoder_sequences.append([word2index['<BOS>']] + conversation[1] + [word2index['<EOS>']])

#Shuffle encoder and decoder rsequences on the same way
c = list(zip(encoder_sequences, decoder_sequences))
random.shuffle(c)
encoder_sequences, decoder_sequences = zip(*c)

#Reduce the number of samples and vobaulary dicts
encoder_sequences = encoder_sequences[:number_sample_conversations]
decoder_sequences = decoder_sequences[:number_sample_conversations]

#Delete unused vocabs
used_words=[]
for sentence in encoder_sequences:
  for id in sentence:
    used_words.append(id)

for sentence in decoder_sequences:
  for id in sentence:
    used_words.append(id)

used_words = set(used_words)

new_index2word = {}
for id in index2word.keys():
  if id in used_words:
    new_index2word[id] = index2word[id]
  
index2word = new_index2word

word2index = {}
for id, word in index2word.items():
  word2index[word] = id

#Summarize the vocab ids to fit max id and lengt and replace the ids in the sentences
#Create temporary mapping
index2counter = {0:0}
word2counter = {}
counter = 1
for word, id in word2index.items():
  index2counter[id] = counter
  word2counter[word] = counter
  counter += 1

  #replace old ids with new ids in sentences
temp_encoder_sequences = []
for sentence in encoder_sequences:
  temp_encoder_sequences.append([index2counter[i] for i in sentence])

encoder_sequences = temp_encoder_sequences

temp_decoder_sequences = []
for sentence in decoder_sequences:
  temp_decoder_sequences.append([index2counter[i] for i in sentence])

decoder_sequences = temp_decoder_sequences

#Update the Vocab dictionaries
word2index = word2counter
index2word = {}
for word, id in word2index.items():
  index2word[id] = word

#Create lists of strings with input and output sentences 
encoder_text = []
for sentence in encoder_sequences:
  encoder_text.append(" ".join([index2word[x] for x in sentence if x != 0]))

decoder_text = []
for sentence in decoder_sequences:
  decoder_text.append(" ".join([index2word[x] for x in sentence if x != 0]))

metadata= {'word2index' : word2index, 'index2word' : index2word, 'encoder_sequences' : encoder_sequences,'decoder_sequences' : decoder_sequences}

#Writes the index lists of message and response and the metadata to file
with open('preprocessed_data.pkl', 'wb') as f:
        pickle.dump(metadata, f)