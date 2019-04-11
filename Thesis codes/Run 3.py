#Depression Detection Model With Python

import nltk
import numpy as np
import random
import copy
from nltk.classify.scikitlearn import SklearnClassifier
import pickle
from nltk.util import ngrams
import bangla_pos_tagger
import collections
from stop_words_remover import remove_stop_words
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
from nltk.metrics.scores import precision, recall, f_measure
from nltk.tokenize import word_tokenize

btagger=bangla_pos_tagger.BanglaTagger()


Depressed = open("P:/Study/Thesis works/Masum405/testfile/Depressed22.txt","r", encoding = 'utf8').read()
Not_Depressed = open("P:/Study/Thesis works/Masum405/testfile/Not Depressed22.txt","r", encoding = 'utf8').read()


# move this up here
all_words = []
documents = []


#N-Grams making function

def get_ngrams(tokens, n ):
    n_grams = ngrams(tokens, n)

    return [ ' '.join(grams) for grams in n_grams]



def doc_ngram(doc):

    token=word_tokenize(doc)

    ln=get_ngrams(token,2)

    return ln


def TF(doc):

    tf_score={}
    scor=0
    total=doc_ngram(doc)
    leng=len(total)
    for  i in total:
        tf_score[i]=total.count(i)/float(leng)

    return tf_score


def IDF(IDF_val,featuresets, documents):


    c=-1
    for j in featuresets:
        c+=1
        for t in j[0]:
            count = 0
            for i in featuresets:
                if t in i[0]:
                    count+=1
            #print(t,' is found in ->', count,' Documents' )

            IDF_val[c][0][t]=np.log(total_doc(documents)/float(count))


    return IDF_val



def total_doc(documents):
    le=len(documents)
    return le




#a=TF('ধন্যবাদ আল্লাহকে আমাকে এমন একটা দুঃখে ভরা জীবন দেবার জন্য বিদায় সবাই')



#def create_freq_dict(doc):



for p in Depressed.split('\n'):
    documents.append( (p, "D") )# tag with D
    #print(documents)
    words = word_tokenize(p)
    #print(words)
    filtered_words = remove_stop_words(words)
    filtered_words=get_ngrams(filtered_words,2)
    #print('filter word = ',filtered_words)
    ## Without D Tag code
    for ii in filtered_words:
          all_words.append(ii)




for p in Not_Depressed.split('\n'):
    documents.append( (p, "ND") )# tag with neg
    words = word_tokenize(p)
    #print(words)
    filtered_words = remove_stop_words(words)

    filtered_words=get_ngrams(filtered_words,2)


    ## Without POS Tag code
    for ii in filtered_words:
          all_words.append(ii)




save_status = open("P:/Study/Thesis works/Masum405/pickle/documents.pickle","wb")

pickle.dump(documents, save_status)
save_status.close()


all_words = nltk.FreqDist(all_words) # FreqDist that means total frequency count
#print("all_words  = "+ str(all_words))
print("all_words length is = "+ str(len(all_words)))



word_features = list(all_words.keys())[: 7175 ]



save_word_features = open("P:/Study/Thesis works/Masum405/pickle/word_features5k.pickle","wb")
pickle.dump(word_features, save_word_features)
save_word_features.close()



#print(find_features('ধন্যবাদ আল্লাহকে আমাকে এমন একটা দুঃখে ভরা জীবন দেবার জন্য বিদায় সবাই'))


#Convert whole document into 2 gramss
featuresets = [(TF(rev), category) for (rev, category) in documents]

temp=copy.deepcopy(featuresets)

IDF_values=IDF(temp,featuresets,documents)


#print(IDF_values)

#print(IDF_values['আম্মু মারা'])

IDF_value=copy.deepcopy(IDF_values)



def TF_IDF(featuresets,IDF_value):
    c=-1
    for i in featuresets:
        c+= 1
        for j in i[0]:
            #print(j)
            #print('TF value is -> ',featuresets[c][0][j] )
            #print('IDF value is -> ', IDF_value[c][0][j])
            featuresets[c][0][j]=featuresets[c][0][j]*IDF_value[c][0][j]
            #print('TF-IDF value is -> ',featuresets[c][0][j])





TF_IDF(featuresets,IDF_value)



print("length before shuffling = "+ str(len(featuresets)))

save_featuresets = open("P:/Study/Thesis works/Masum405/pickle/featuresets.pickle","wb")
pickle.dump(featuresets, save_featuresets)
save_featuresets.close()

random.shuffle(featuresets)


print("length after shuffling = "+ str(len(featuresets)))


training_set = featuresets[: 514 ]
testing_set = featuresets[ 514 :]


# print('training_set = ',str(training_set))# 'নিউজিল্যাণ্ড': False,'দল': True,
# print('testing_set = ',str(testing_set)) #

NB_classifier = nltk.NaiveBayesClassifier.train(training_set)
refsets = collections.defaultdict(set)
testsets = collections.defaultdict(set)

for i, (feats, label) in enumerate(testing_set):
    # print('i = ',str(i),' feats = ',str(feats))
    # print('i = ', str(i), ' label = ', str(label))# label= D, neg
    refsets[label].add(i)
    # print(str(refsets[label]))
    observed = NB_classifier.classify(feats)
    # print('observed = ',observed)# observed= D, neg
    testsets[observed].add(i)
    # print( str(testsets[observed]))

# print('refsets = ',str(refsets))
# print('testsets = ',str(testsets))

print("for bi-gram\n")
print("Original Naive Bayes classifier accuracy:", (nltk.classify.accuracy(NB_classifier, testing_set)) * 100)

# print('confusion matrix')
# print((confusion_matrix(refsets['neg'], testsets['neg']))*100)

print('D precision:', (precision(refsets['D'], testsets['D'])) * 100)
print('D recall:', (recall(refsets['D'], testsets['D'])) * 100)
print('D f_measure:', (f_measure(refsets['D'], testsets['D'])) * 100)
print('ND precision:', (precision(refsets['ND'], testsets['ND'])) * 100)
print('ND recall:', (recall(refsets['ND'], testsets['ND'])) * 100)
print('ND f_measure:', (f_measure(refsets['ND'], testsets['ND'])) * 100)

NB_classifier.show_most_informative_features(10)

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)

for i, (feats, label) in enumerate(testing_set):
    refsets[label].add(i)
    observed = MNB_classifier.classify(feats)
    testsets[observed].add(i)
print("\n")
print("MNB_classifier accuracy percent:", (nltk.classify.accuracy(MNB_classifier, testing_set)) * 100)
print('D precision:', precision(refsets['D'], testsets['D']))
print('D recall:', recall(refsets['D'], testsets['D']))
print('D f_measure:', f_measure(refsets['D'], testsets['D']))
print('ND precision:', precision(refsets['ND'], testsets['ND']))
print('ND recall:', recall(refsets['ND'], testsets['ND']))
print('ND f_measure:', f_measure(refsets['ND'], testsets['ND']))

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)

for i, (feats, label) in enumerate(testing_set):
    refsets[label].add(i)
    observed = LogisticRegression_classifier.classify(feats)
    testsets[observed].add(i)
print("\n")
print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set)) * 100)
print('D precision:', (precision(refsets['D'], testsets['D'])) * 100)
print('D recall:', (recall(refsets['D'], testsets['D'])) * 100)
print('D f_measure:', (f_measure(refsets['D'], testsets['D'])) * 100)
print('ND precision:', (precision(refsets['ND'], testsets['ND'])) * 100)
print('ND recall:', (recall(refsets['ND'], testsets['ND'])) * 100)
print('ND f_measure:', (f_measure(refsets['ND'], testsets['ND'])) * 100)


LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)

for i, (feats, label) in enumerate(testing_set):
    refsets[label].add(i)
    observed = LinearSVC_classifier.classify(feats)
    testsets[observed].add(i)
print("\n")
print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set)) * 100)
print('D precision:', (precision(refsets['D'], testsets['D'])) * 100)
print('D recall:', (recall(refsets['D'], testsets['D'])) * 100)
print('D f_measure:', (f_measure(refsets['D'], testsets['D'])) * 100)
print('ND precision:', (precision(refsets['ND'], testsets['ND'])) * 100)
print('ND recall:', (recall(refsets['ND'], testsets['ND'])) * 100)
print('ND f_measure:', (f_measure(refsets['ND'], testsets['ND'])) * 100)

NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)

for i, (feats, label) in enumerate(testing_set):
    refsets[label].add(i)
    observed = NuSVC_classifier.classify(feats)
    testsets[observed].add(i)
print("\n")
print("NuSVC_classifier accuracy percent:", (nltk.classify.accuracy(NuSVC_classifier, testing_set)) * 100)
print('D precision:', (precision(refsets['D'], testsets['D'])) * 100)
print('D recall:', (recall(refsets['D'], testsets['D'])) * 100)
print('D f_measure:', (f_measure(refsets['D'], testsets['D'])) * 100)
print('ND precision:', (precision(refsets['ND'], testsets['ND'])) * 100)
print('ND recall:', (recall(refsets['ND'], testsets['ND'])) * 100)
print('ND f_measure:', (f_measure(refsets['ND'], testsets['ND'])) * 100)

BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)

for i, (feats, label) in enumerate(testing_set):
    refsets[label].add(i)
    observed = BernoulliNB_classifier.classify(feats)
    testsets[observed].add(i)
print("\n")
print("BernoulliNB_classifier accuracy percent:", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set)) * 100)
print('D precision:', (precision(refsets['D'], testsets['D'])) * 100)
print('D recall:', (recall(refsets['D'], testsets['D'])) * 100)
print('D f_measure:', (f_measure(refsets['D'], testsets['D'])) * 100)
print('ND precision:', (precision(refsets['ND'], testsets['ND'])) * 100)
print('ND recall:', (recall(refsets['ND'], testsets['ND'])) * 100)
print('ND f_measure:', (f_measure(refsets['ND'], testsets['ND'])) * 100)

SGDC_classifier = SklearnClassifier(SGDClassifier())
SGDC_classifier.train(training_set)

for i, (feats, label) in enumerate(testing_set):
    refsets[label].add(i)
    observed = SGDC_classifier.classify(feats)
    testsets[observed].add(i)
print("\n")
print("SGDC_classifier accuracy percent:", (nltk.classify.accuracy(SGDC_classifier, testing_set)) * 100)
print('D precision:', (precision(refsets['D'], testsets['D'])) * 100)
print('D recall:', (recall(refsets['D'], testsets['D'])) * 100)
print('D f_measure:', (f_measure(refsets['D'], testsets['D'])) * 100)
print('ND precision:', (precision(refsets['ND'], testsets['ND'])) * 100)
print('ND recall:', (recall(refsets['ND'], testsets['ND'])) * 100)
print('ND f_measure:', (f_measure(refsets['ND'], testsets['ND'])) * 100)

print('\n\nIs it Ok run-2 file??')