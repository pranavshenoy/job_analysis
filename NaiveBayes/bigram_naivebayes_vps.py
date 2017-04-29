import nltk
import random
import os
import json
import thread
import time
import pickle

def load_dataset():  
    dataset=[]
    for root, dirs, files in os.walk('../Dataset'):
        for directory in dirs:
            for root_f, dirs_f, files_f in os.walk(os.path.join(root,directory)):
                for each_file in files_f:
                    with open(os.path.join(root,directory)+'/'+each_file) as f:
                        data_temp=json.load(f)    
                    dataset+=[(sentence,each_file) for sentence in data_temp]
    return dataset

def get_words(x):    #returns all the words in the  data set
    vocab = []
    for lists,polarity in x:
        for each_item in lists:
            vocab.append(each_item[0]+' '+each_item[1])
    return vocab

#execution
def execute():
    labeled_data=load_dataset()
    return labeled_data

labeled_data=execute()

vocab=[]
with open('vocab_bigram') as f:
	vocab=json.load(f)

def extract_features(document):       #each sentences contains a list of words and true/false     input-sentence
    features = {}
    document=[items[0]+' '+items[1] for items in nltk.bigrams(document.split())]
    for word in vocab:
         features['contains(%s)' % word] = (word in document)
    return features        

train_test = nltk.classify.apply_features(extract_features, labeled_data)

train_set=train_test[:-10000]
test_set=train_test[-10000:]

print 'started training dataset'
classifier = nltk.NaiveBayesClassifier.train(train_set)
#save  the model
with open('naive_bayes_bigram_model','w') as f:
    pickle.dump(classifier,f)
print 'completed training dataset'

print 'started computing accuracy'
#accuracy
nltk.classify.accuracy(classifier, test_set)
#save accuracy
with open('accuracy_bigram_model','w') as f:
    pickle.dump(classifier,f)
print 'completed computing accuracy'


