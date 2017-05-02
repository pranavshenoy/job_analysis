import nltk
import random
import os
import json
from nltk.stem import  PorterStemmer
import random
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

#initialising porterstemmer
ps=PorterStemmer()

def to_unigram(data):
    dataset=[]
    dataset+=[([items for items in sentence.split()],polarity) for sentence,polarity in data]    
    return dataset    

def get_words(x):    #returns all the words in the  data set
    vocab = []
    for sentence,polarity in x:  
        vocab.extend(sentence.split())
    return vocab

#execution
labeled_data=load_dataset()    #loading labeled sentences
random.shuffle(labeled_data)
vocab=set(get_words(labeled_data)) #getting words 
vocab=set([ps.stem(word) for word in vocab])   #stemming
#with open('vocab_unigram','w') as f:
#    json.dump(vocab,f)
#labeled_data=to_unigram(labeled_data)    #converting to unigram

def extract_features(document):       #features are bag of words. document is a list of words of a sentence 
    features = {}
    for word in vocab:
        features['contains(%s)' % word] = (word in document)
    return features        

train_test = nltk.classify.apply_features(extract_features, labeled_data)

#dividing into training and test data
train_set=train_test[:-10000]
test_set=train_test[-2000:]

with open('naive_bayes_unigram_model') as f:
    classifier=pickle.load(f)

accuracy=nltk.classify.accuracy(classifier, test_set)

with open('accuracy_unigram_model','w') as f:
    pickle.dump(accuracy,f)   
                