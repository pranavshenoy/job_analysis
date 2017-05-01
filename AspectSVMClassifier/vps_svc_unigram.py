import random
import os
from sklearn import svm
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, f1_score
import pickle

label_map={'Aspect1':1 ,'Aspect2':2 ,'Aspect3':3 ,'Aspect4':4 ,'Aspect5':5,'Aspect6':6 ,'Aspect7':7}

def load_dataset():  
    dataset=[]
    for root, dirs, files in os.walk('../Dataset'):
        for directory in dirs:
            for root_f, dirs_f, files_f in os.walk(os.path.join(root,directory)):
                for each_file in files_f:
                    with open(os.path.join(root,directory)+'/'+each_file) as f:
                        data_temp=json.load(f)    
                    dataset+=[(sentence,label_map[directory]) for sentence in data_temp]
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

print 'Dataset ready'
vectorizer = CountVectorizer(vocabulary=vocab,min_df=1)
corpus=[sentence for sentence,pol in labeled_data]
features=vectorizer.fit_transform(corpus[:50000])
labels=[label for sentence,label in labeled_data[:50000]]

print 'feature extraction completed'
classifier=svm.SVC()
classifier.kernel='linear'
classifier.fit(features[:30000],labels[:30000])
with open('svc_unigram_model','w') as f:
    pickle.dump(classifier,f) 
print 'training completed'

print 'Accuracy Computing started'
predicted = classifier.predict(features[35000:40000])
accuracy=classifier.accuracy_score(labels[35000:40000],predicted)
with open('accuracy_svc_unigram','w') as f:
    pickle.dump(classifier,f) 
print 'Accuracy Computing completed'


