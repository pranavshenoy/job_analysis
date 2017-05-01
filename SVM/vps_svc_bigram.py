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

def to_bigram(data):
    dataset=[]
    dataset+=[([items for items in nltk.bigrams(sentence.split())],polarity) for sentence,polarity in data]    
    return dataset        

#execution
labeled_data=load_dataset()    #loading labeled sentences
random.shuffle(labeled_data)
with open('../NaiveBayes/vocab_bigram') as f:
    vocab=json.load(f)    

def extract_features(dataset):       #input a  sentence ...output bigrams
    feature_vector=[]
    labels=[]
    for sentence,label in dataset:
        features = {}
        labels.append(label)
        sentence=[items[0]+' '+items[1] for items in nltk.bigrams(sentence.split())]
        for word in vocab:
            features[word] = 0
            if word in sentence:
                features[word] = 1
        feature_vector.append(features.values())        
    return {'feature':feature_vector,'labels':labels}        
print 'Dataset started'                   
features=extract_features(labeled_data[:20000]) 
print 'dataset Completed'                   
classifier=svm.SVC()
print 'Training Started'                   
classifier.fit(features['feature'],features['labels'])
with open('svc_bigram_model','w') as f:
    pickle.dump(classifier,f)

print 'Training Completed'                   