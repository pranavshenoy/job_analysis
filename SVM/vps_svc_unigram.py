import random
import os
from sklearn import svm
import json
from sklearn.feature_extraction.text import CountVectorizer
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

#execution
labeled_data=load_dataset()    #loading labeled sentences
random.shuffle(labeled_data)

print 'Dataset ready'
vectorizer = CountVectorizer(min_df=1)
corpus=[sentence for sentence,pol in labeled_data]
features=vectorizer.fit_transform(corpus[:50000])
labels=[label for sentence,label in labeled_data[:50000]]

print 'feature extraction completed'
classifier=svm.SVC()
classifier.fit(features,labels)
with open('svc_unigram_model','w') as f:
    pickle.dump(classifier,f) 
print 'training completed'

               
