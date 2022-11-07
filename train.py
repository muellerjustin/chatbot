# import required frameworks and libraries
import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

import json
import pickle
import numpy as np
import random

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD

lemmatizer = WordNetLemmatizer()

# create words, tags and documents
words = []
tags = []
documents = []

# create list which includes characters which should be ignored
ignore_words = [".", ",", ";", "?", "!", "-"]

# open and load intents
data = open('intents.json').read()
intents = json.loads(data)

# loop through intents
for intent in intents['intents']:
    # loop through patterns
    for pattern in intent['patterns']:
        # tokenize each pattern - split into single words
        # example of tokenization: "How are you?" -> ["How", "are", "you", "?"]
        w = nltk.word_tokenize(pattern)

        # add words to words
        words.extend(w)

        # add documents to documents
        # example of a document: (["How", "are", "you", "?"], "greeting")
        documents.append((w, intent['tag']))

        # add tags to tags
        # examples of tag: "greeting", "location", "opening hours"
        if intent['tag'] not in tags:
            tags.append(intent['tag'])


# lemmatize each word: create base word to represent related words
# example of lemmatization: goes, go, going, gone -> go
# lowercase and check if word is in ignore_words
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]

# create sorted list and delete identical words and tags
words = sorted(list(set(words)))
tags = sorted(list(set(tags)))

# convert words and tags into pickle files 
pickle.dump(words,open('words.pkl','wb'))
pickle.dump(tags,open('tags.pkl','wb'))

# create training and output with length of tags
training = []
output_empty = [0] * len(tags)

# loop through documents
for doc in documents:
    # create bag of words
    bag = []

    # list of tokenized words
    # example: ["What", "are", "the", "opening", "hours", "?"]
    pattern_words = doc[0]
    # lemmatize each word: create base word to represent related words
    # lowercase words
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]

    # loop through words
    for w in words:
        # append 1 to bag if word in words matches word in pattern_words otherwise append 0
        # example:
        # words = ["hello", "hi", "hey", "how", "are", "you", "there"]
        # pattern_words = ["hi", "there"]
        # -> bag = [0, 1, 0, 0, 0, 0, 1]
        bag.append(1) if w in pattern_words else bag.append(0)
    
    # for each tag output is 0 and for current tag output is 1
    # example:
    # tags = ["greeting", "products", "contact", "location", "opening hours", "thanks"]
    # doc[1] = "contact"
    # -> output_row = [0, 0, 1, 0, 0, 0]
    output_row = list(output_empty)
    output_row[tags.index(doc[1])] = 1

    # append bag and output_row to training
    training.append([bag, output_row])
    

# shuffle training and convert into np array
random.shuffle(training)
training = np.array(training)

# create train_x (patterns) and train_y (tags)
# example:
# training = [[[0, 0, 1], [0, 1, 0]], [[1, 0, 0], [1, 1, 0]], [[1, 1, 1], [0, 1, 1]]]
# train_x = [[0, 0, 1], [1, 0, 0], [1, 1, 1]]
# train_y = [[0, 1, 0], [1, 1, 0], [0, 1, 1]]
train_x = list(training[:,0])
train_y = list(training[:,1])

# create model with 3 layers
# input layer with 128 neurons
# hidden layer with 64 neurons
# output layer with number of neurons equal to number of possible intents
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# compile model
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# fit and save model
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)
print("model created")
