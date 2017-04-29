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
                    dataset+=[(sentence,directory) for sentence in data_temp]
    return dataset

#initialising porterstemmer
ps=PorterStemmer()

def to_unigram(data):
    dataset=[]
    dataset+=[([items for items in sentence.split()],aspect) for sentence,aspect in data]    
    return dataset    

def get_words(x):    #returns all the words in the  data set
    vocab = []
    for sentence,polarity in x:  
        vocab.extend(sentence.split())
    return vocab

print 'Execution Started'
#execution
labeled_data=load_dataset()    #loading labeled sentences
random.shuffle(labeled_data)

vocab=set(get_words(labeled_data)) #getting words 
vocab=[ps.stem(word) for word in vocab]   #stemming
#with open('vocab_unigram','w') as f:
#    json.dump(vocab,f)
labeled_data=to_unigram(labeled_data)    #converting to unigram

print 'Vocab and Dataset Ready'

def extract_features(document):       #features are bag of words. document is a list of words of a sentence 
    features = {}
    for word in vocab:
        features['contains(%s)' % word] = (word in document)
    return features        

train_test = nltk.classify.apply_features(extract_features, labeled_data)

#dividing into training and test data
train_set=train_test[:-10000]
test_set=train_test[-10000:]

print 'Training Started'

classifier = nltk.NaiveBayesClassifier.train(train_set)
with open('naive_bayes_unigram_model','w') as f:
    pickle.dump(classifier,f)

print 'Training Completed'
print 'Accuracy Started'

accuracy=nltk.classify.accuracy(classifier, test_set)
with open('accuracy','w') as f:
    json.dump(accuracy,f)                
print 'Accuracy Completed'
