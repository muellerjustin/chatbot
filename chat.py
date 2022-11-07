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

from keras.models import load_model

lemmatizer = WordNetLemmatizer()

# load intents, words, tags, model
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl','rb'))
tags = pickle.load(open('tags.pkl','rb'))
model = load_model('chatbot_model.h5')

# tokenize and lemmatize sentence
def tokenize_and_lemmatize(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


# create bag of words array
def bow(sentence, words):
    # pass sentence into tokenize_and_lemmatize function
    sentence_words = tokenize_and_lemmatize(sentence)

    # create bag with length of words
    bag = [0]*len(words)

    # loop through words in tokenized and lemmatized sentence
    for s in sentence_words:
        # loop through words with related index in words
        for i,w in enumerate(words):
            # if word in sentence is identical to word in words assign 1 to index of current word
            if w == s:
                bag[i] = 1

    return(np.array(bag))


def predict_intent(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words)

    # probabilities of each intent
    res = model.predict(np.array([p]))[0]

    # probability of intent must be at least 75% 
    ERROR_THRESHOLD = 0.75

    # add intent with related index to results if probability is above treshold
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]

    # sort results by strength of probability of intent
    results.sort(key=lambda x: x[1], reverse=True)

    # create return_list
    return_list = []

    # loop through results in results
    for r in results:
        # append intent category and its probability to return_list
        return_list.append({"intent": tags[r[0]], "probability": str(r[1])})

    # if there is no element in return_list because probability of intents is below treshold
    # then append intent "noanswer" to return_list
    if not return_list:
        return_list.append({"intent": "noanswer", "probability": "1"})

    return return_list
    

def getResponse(ints, intents_json):
    # get tag
    tag = ints[0]['intent'] # (*)

    list_of_intents = intents_json['intents']
    # loop trough list_of_intents
    # example for an element of list_of_intents: {'tag': 'greeting', 'patterns': ['Hi there', 'How are you',"What's up?"], 'responses': ['Hello there', 'Good to see you again']}
    for i in list_of_intents:

        # if tag in list_of_intents is identical to tag (*), then give random response related to this intent
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
        
    return result


# take message of user and return response
def chatbot_response(msg):
    # get intent (*)
    ints = predict_intent(msg, model)

    # get response related to intent (*)
    res = getResponse(ints, intents)

    return res


if __name__ == "__main__":
    # start chat with short message
    print("Let's chat! (type 'quit' to exit)")

    # start infinite loop
    while True:
        # get message from user
        sentence = input("You: ")

        # if user types in "quit" stop chat
        if sentence == "quit":
            break
        
        # pass in user input to chatbot_response function
        resp = chatbot_response(sentence)

        # print response from chatbot to chat in terminal
        print(resp)