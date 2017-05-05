import nltk
import random
import os
import json
from nltk.stem import  PorterStemmer
import random
from nltk.corpus import wordnet as wn
import pickle


def load_dataset():  
    dataset=[]
    for root, dirs, files in os.walk('../Dataset_new'):
        for directory in dirs:
            for root_f, dirs_f, files_f in os.walk(os.path.join(root,directory)):
                for each_file in files_f:
                    with open(os.path.join(root,directory)+'/'+each_file) as f:
                        data_temp=json.load(f)    
                    dataset+=[sentence for sentence in data_temp]
    return dataset


dataset=load_dataset()
vocab=[word for sentence in dataset for word in sentence.split()]
counts=[(x,vocab.count(x)) for x in vocab]
with open('analyse_count_words','w') as f:
    json.dump(counts,f)    