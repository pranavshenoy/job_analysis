import nltk
import random
import os
from sklearn import svm
import json
from nltk.stem import  PorterStemmer
import random
import pickle

label_map={'neg':0 ,'sli_neg':1 ,'neutral':2 ,'sli_pos':3 ,'pos':4}

def load_dataset():  
    dataset=[]
    for root, dirs, files in os.walk('../Dataset'):
        for directory in dirs:
            for root_f, dirs_f, files_f in os.walk(os.path.join(root,directory)):
                for each_file in files_f:
                    with open(os.path.join(root,directory)+'/'+each_file) as f:
                        data_temp=json.load(f)    
                    dataset+=[(sentence,label_map[each_file]) for sentence in data_temp]
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
vocab=[ps.stem(word) for word in vocab]   #stemming
#with open('vocab_unigram','w') as f:
#    json.dump(vocab,f)
labeled_data=to_unigram(labeled_data)    #converting to unigram    

print 'Dataset ready'
def extract_features(dataset):       #features are bag of words. document is a list of words of a sentence 
    feature_vector=[]
    labels=[]
    for sentence,label in dataset:
        features = {}
        labels.append(label)
        for word in vocab:
            features[word] = 0
            if word in sentence:
                features[word] = 1
        feature_vector.append(features.values())        
    return {'feature':feature_vector,'labels':labels}        

features=extract_features(labeled_data)
print 'feature extraction completed'
classifier=svm.SVC()
classifier.fit(features['feature'],features['labels'])
with open('svc_unigram_model','w') as f:
    pickle.dump(classifier,f) 
print 'training completed'

               