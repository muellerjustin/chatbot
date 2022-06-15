# import required libraries
import json
import random 
import nltk
import pickle
import numpy as np
from nltk.stem import WordNetLemmatizer 
from keras import Sequential 
from keras.layers import Dense, Dropout
from keras.optimizers import SGD

nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()

## load data from the json file as a dictionary
# tag: category
# pattern: examples of what a user input could look like 
# responses: hard coded answers the chatbot can give
intents = json.loads(open("intents.json").read())

# create lists
words = []
classes = []
documents = []
ignore_characters = [".", ",", "!", "?"]

# loop through intents list
for intent in intents["intents"]:
    # loop through patterns list
    for pattern in intent["patterns"]:
        # split up pattern into single words and append to to words list
        # "Wie geht es dir?" -> ["Wie", "geht", "es", "dir", "?"]
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)

        # append tuple of list of single words and their tag
        documents.append((word_list, intent["tag"]))

        # check if tag is in classes list
        # if not, append tag to classes list
        if intent["tag"] not in classes:
            classes.append(intent["tag"])


# create new_words list         
new_words = []

# loop through words list
for word in words:
    # ignore the characters defined above
    if word not in ignore_characters:
        # convert word to its base form and append to new_words list
        new_word = lemmatizer.lemmatize(word)
        new_words.append(new_word)

# remove duplicates, arrange in alphabetical order and words beginning with capital letters first 
new_words = sorted(set(new_words))
classes = sorted(set(classes))

# save words list and classes list in pickle files
pickle.dump(new_words, open("words.pkl", "wb"))
pickle.dump(classes, open("classes.pkl", "wb"))


## convert words to numerical values in order to feed the neural network
# create training list
training = []
# create output_empty list which contains as many zeros as classes list has elements
output_empty = [0] * len(classes)

# create new_word_patterns list
new_word_patterns = []

# loop through documents list
for document in documents:
    # create bag list
    bag = []

    # get first element of document tuple which is a list
    # --> document = (["What", "is", "your", "name", "?"], "name")
    word_patterns = document[0]
    
    # loop through word_patterns list
    for word in word_patterns:
        # convert word to its base form and append to new_words_patterns list
        newWord = lemmatizer.lemmatize(word.lower())
        new_word_patterns.append(newWord)

    # loop through new_words list
    for word in new_words:
        # check if word is in new_word_patterns list
        # if so, append number 1 to bag list
        # else append number 0 to bag list
        if word in new_word_patterns:
            bag.append(1)
        else:
            bag.append(0)

    # copy output_empty list
    output_row = list(output_empty)

    output_row[classes.index(document[1])] = 1
    # append bag list and output_row list to training list
    training.append([bag, output_row])


# reorganize items of training list randomly
random.shuffle(training)
# turn training list into a numpy array
training = np.array(training)

# split training list into x and y values
# complete list and zeroth dimension
train_x = list(training[:, 0])
# compelte list and first dimension
train_y = list(training[:, 1])


