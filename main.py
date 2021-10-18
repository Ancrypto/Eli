#Eli
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random
import json
import pickle
import os
import colorama
from colorama import Fore, Style, Back
import shutil

columns = shutil.get_terminal_size().columns

with open("Brain/JSON/intents.json") as file:
    data = json.load(file)


try:
    with open('Brain/Resource/data.pickle', "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

            if intent["tag"] not in labels:
                labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = numpy.array(training)
    output = numpy.array(output)

    with open('Brain/Resource/data.pickle', "wb") as f:
        pickle.dump((words, labels, training, output), f)


tensorflow.compat.v1.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

eli = tflearn.DNN(net)

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    
    return numpy.array(bag)


def chat():
    print("")
    print("Hi I am Eli!")
    while True:
        inp = input(">> ")
        if inp.lower() == "exit":
            break

        results = eli.predict([bag_of_words(inp, words)])
        results_index = numpy.argmax(results)
        tag = labels[results_index]

        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']
        
        print(random.choice(responses))

def Header():
    os.system('clear')
    print (Fore.BLUE + Style.BRIGHT + "_/-- Eli --\_".center(columns))
    print(Style.RESET_ALL)
    print("")

def main():
    #main function

    Header()

    print("Use 'list' to list commands...")

    while True:
        inp = input("Console: ")
        if inp.lower() == "exit":
            break

        #All commands
        #list all commands
        elif(inp == "list"):
            print("train -- retrains eli")
            print("chat -- chat with eli")
            print("load -- load eli")

            print("exit -- quit the console") 
            print("")
            inp = ""
        #retrain eli
        elif(inp == "train"):
            eli.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
            eli.save("Brain/Resource/Eli.tflearn")
            Header()

            inp = ""
        #chat with eli
        elif(inp == "chat"):
            chat()
            Header()
            
            inp = ""
        #load eli brain files
        elif(inp == "load"):
            eli.load("Brain/Resource/./Eli.tflearn")
            Header()

            inp = ""
        #resetting eli's brain
        elif(inp == "reset"):
            eli.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
            eli.save("Brain/Resource/Eli.tflearn")
            Header()

            inp = ""
        else:
            print("Command '"+inp+"' not found!")
            print("")
            
            inp = ""

main()