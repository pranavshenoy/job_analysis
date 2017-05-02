import re
import nltk
import os
import random
import json
import pickle
import numpy as np
from nltk.stem import PorterStemmer
from sklearn import svm
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
with open('Dataset/Samsung.txt') as f:
    dataset=json.load(f)
ps=PorterStemmer()


#bigram svc
with open('../NaiveBayes/vocab_bigram') as f:
    vocab_svc_bi=json.load(f)   
def get_feature_svc_bi(sentence):
    feature=[]
    document=[items[0]+' '+items[1] for items in nltk.bigrams(sentence.split())]
    for word in vocab_svc_bi: 
        if word in document:
            feature.append(1)
        else:
            feature.append(0)
    return feature        

#fetching aspects
aspects={}
for root, dirs, files in os.walk('../Aspects/'):
        for name in files:
            with open('../Aspects/'+name) as f:
                aspects[name]=json.load(f)          #loading pool of words for each aspects
                    
def analyse_aspect(tokens):            # analyses to which aspect a set of tokens of a sentence belong to
    count=0
    #print 'Tokens : '+ str(tokens)
    aspects_in_sentence=[]
    for token in tokens:
        temp_token=ps.stem(token)
        for aspect_name in files:                               #each aspect
            for aspect in aspects[aspect_name]:                 #each word in an aspect
                temp_aspect=ps.stem(aspect)
                if(temp_token == temp_aspect):
                    aspects_in_sentence.append(aspect_name)
                    count=1
                    break
    if(count==0):
        aspects_in_sentence.append('Aspect7')
    #print 'List Of Aspects : '+ str(aspects_in_sentence)
    return   aspects_in_sentence          #aspect names in which it belongs

#loading vocab for navebayes unigram
with open('../NaiveBayes/vocab_unigram') as f:
    vocab_nb_uni=json.load(f)
def extract_features_unigram_nb(document):       #features are bag of words. document is a list of words of a sentence 
    features = {}
    for word in vocab_nb_uni:
        features['contains(%s)' % word] = (word in document)
    return features
        
#aspect classification using naive bayes
def analyse_aspect_naivebayes(words):       
    aspects_in_sentence=[]
    with open('../AspectNaiveBayesClassifier/naive_bayes_unigram_model') as f:
        classifier=pickle.load(f)
    aspects_in_sentence.append(classifier.classify(extract_features_unigram_nb(words)))
    return aspects_in_sentence


#svm bigram
def svc_bigram(dataset,aspect_analysis):        #first find the polarity and then its aspect
    #loading naivebayes classifier
    with open('../SVM/svc_bigram_model') as f:
        classifier=pickle.load(f)    
    aspect_polarity=Counter()
    aspect_count=Counter()
    for sentence in dataset:
        #print ' '
        #print 'sentence : '+sentence
        polarity=0
        flag=0       #stores label number
        feature=get_feature_svc_bi(sentence)
        flag=classifier.predict(feature)
        #print 'Class= '+ str(flag)        
        polarity=classifier.predict_proba(feature)[0][flag]
        #print 'Polarity = '+str(polarity)
        if(flag==2):         #Normalising
            polarity=0
        elif(flag==4):
            polarity=+0.5+(polarity/2)
        elif(flag==0):
            polarity=-0.5-(polarity/2)    
        elif(flag==3):
            polarity=polarity/2
        elif(flag==1):
            polarity=-(polarity/2)   
        #print 'final polarity='+ str(polarity)
        #print ' '
        if('neutral'!= flag):
            if aspect_analysis == 'lexical':
                aspects_in_sentence=analyse_aspect(sentence.split())    # returns list of aspects where polarity should be added
            elif aspect_analysis == 'naivebayes':
                aspects_in_sentence=analyse_aspect_naivebayes(sentence.split())
            for asp in aspects_in_sentence:
                aspect_polarity[asp]=aspect_polarity[asp]+polarity
                aspect_count[asp]=aspect_count[asp]+1
            #print 'aspect_polarity ='+str(aspect_polarity)
            #print ' '
            #print 'aspect_count ='+str(aspect_count)
        #print '---------------------------'
    return aspect_polarity,aspect_count    





#print 'Aspect Classifier = Lexical'
#aspect_polarity7,aspect_count7=svc_bigram(dataset[:1000],'lexical')
#with open('aspect_polarity7','w') as f:
#    pickle.dump(aspect_polarity7,f) 
#with open('aspect_count7','w') as f:
#    pickle.dump(aspect_count7,f)
#print 'Aspect Classifier = Lexical  Completed'

print 'Aspect Classifier = NaiveBayes'
aspect_polarity8,aspect_count8=svc_bigram(dataset[:1000],'naivebayes')
with open('aspect_polarity8','w') as f:
    pickle.dump(aspect_polarity8,f) 
with open('aspect_count8','w') as f:
    pickle.dump(aspect_count8,f)
print 'Aspect Classifier = naivebayes  Completed'
