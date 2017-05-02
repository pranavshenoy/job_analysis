with open('Dataset/Samsung.txt') as f:
    dataset=json.load(f)

#loading vocab for navebayes bigram
with open('../NaiveBayes/vocab_bigram') as f:
    vocab_nb_bi=json.load(f)
def extract_features_bigram_nb(document):       #input is sentence
    features = {}
    document=[items[0]+' '+items[1] for items in nltk.bigrams(document.split())]
    for word in vocab_nb_bi:
         features['contains(%s)' % word] = (word in document)
    return features

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


#aspect classification using naive bayes
def analyse_aspect_naivebayes(words):       
    aspects_in_sentence=[]
    with open('../AspectNaiveBayesClassifier/naive_bayes_unigram_model') as f:
        classifier=pickle.load(f)
    aspects_in_sentence.append(classifier.classify(extract_features_unigram_nb(words)))
    return aspects_in_sentence

#aspect classification using SVC                          #pending
with open('../AspectSVMClassifier/vocab_unigram') as f:
    vocab_svc_uni=json.load(f)
vectorizer = CountVectorizer(vocabulary=vocab_svc_uni,min_df=1)
    
def analyse_aspect_svc(sentence):           # inputs are words for a sentence     
    aspects_in_sentence=[]
    with open('../AspectSVMClassifier/svc_unigram_model') as f:
        classifier=pickle.load(f)
    feature_svc=vectorizer.fit_transform(sentence)    
    predicted_class=classifier.predict(feature_svc)
    #print predicted_class
    #aspects_in_sentence.append()          ######################3
    return aspects_in_sentence
#analyse_aspect_svc('a lot of gym')


#naiveBayes
def naiveBayes_unigram(dataset,aspect_analysis):        #first find the polarity and then its aspect
    #loading naivebayes classifier
    with open('../NaiveBayes/naive_bayes_unigram_model') as f:
        classifier=pickle.load(f)    
    aspect_polarity=Counter()
    aspect_count=Counter()
    for sentence in dataset:
        #print ' '
        #print 'sentence : '+sentence
        #print 'Polarity= '+classifier.classify(extract_features_unigram_nb(sentence.split()))
        polarity=0
        flag=0        
        dist = classifier.prob_classify(extract_features_unigram_nb(sentence.split()))
        for label in dist.samples():
            #print(" %s: %f" % (label, dist.prob(label)))
            if(polarity<dist.prob(label)):
                polarity=dist.prob(label)
                flag=label
        if(flag=='neutral'):         #Normalising
            polarity=0
        elif(flag=='pos'):
            polarity=+0.5+(polarity/2)
        elif(flag=='neg'):
            polarity=-0.5-(polarity/2)    
        elif(flag=='sli_pos'):
            polarity=polarity/2
        elif(flag=='sli_neg'):
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

#naiveBayes
def naiveBayes_bigram(dataset,aspect_analysis):        #first find the polarity and then its aspect
    #loading naivebayes classifier
    with open('../NaiveBayes/naive_bayes_bigram_model') as f:
        classifier=pickle.load(f)    
    aspect_polarity=Counter()
    aspect_count=Counter()
    for sentence in dataset:
        #print ' '
        #print 'sentence : '+sentence
        #print 'Polarity= '+classifier.classify(extract_features_bigram_nb(sentence))
        polarity=0
        flag=0        
        dist = classifier.prob_classify(extract_features_bigram_nb(sentence))
        for label in dist.samples():
            print(" %s: %f" % (label, dist.prob(label)))
            if(polarity<dist.prob(label)):
                polarity=dist.prob(label)
                flag=label
        if(flag=='neutral'):         #Normalising
            polarity=0
        elif(flag=='pos'):
            polarity=+0.5+(polarity/2)
        elif(flag=='neg'):
            polarity=-0.5-(polarity/2)    
        elif(flag=='sli_pos'):
            polarity=polarity/2
        elif(flag=='sli_neg'):
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

aspect_polarity3,aspect_count3=naiveBayes_bigram(dataset[:1000],'naivebayes')
with open('aspect_polarity3','w') as f:
	json.dump(aspect_polarity3,f) 
with open('aspect_count3','w') as f:
	json.dump(aspect_count3,f)	                    
aspect_polarity4,aspect_count4=naiveBayes_bigram(dataset[:1000],'lexical')
with open('aspect_polarity4','w') as f:
	json.dump(aspect_polarity4,f) 
with open('aspect_count4','w') as f:
	json.dump(aspect_count4,f)
