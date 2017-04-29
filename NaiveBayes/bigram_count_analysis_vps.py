import nltk
import random
import os
import json
import thread
import time
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

def to_bigram(data):
    dataset=[]
    dataset+=[([items for items in nltk.bigrams(sentence.split())],polarity) for sentence,polarity in data]    
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
    labeled_data=to_bigram(labeled_data)
    vocab=get_words(labeled_data)
    return labeled_data,vocab

#thread.start_new_thread(execute,())
labeled_data,vocab=execute()
print('vocab fetched completely')
vocab_count=[(x,vocab.count(x)) for x in set(vocab)]
with open('vocab_bigram_count','w') as f:
    json.dump(vocab_count,f)

print 'execution complete'    
