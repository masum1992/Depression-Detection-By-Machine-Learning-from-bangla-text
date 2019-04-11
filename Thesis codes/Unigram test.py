import nltk
import random
import numpy as np
from nltk.classify.scikitlearn import SklearnClassifier
import pickle

import collections
from nltk.util import ngrams
from stop_words_remover import remove_stop_words
from nltk.metrics.scores import precision, recall, f_measure
import copy
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC


from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize



Depressed = open("P:/Study/Thesis works/Masum405/testfile/Depressed22.txt","r", encoding = 'utf8').read()
Not_Depressed = open("P:/Study/Thesis works/Masum405/testfile/Not Depressed22.txt","r", encoding = 'utf8').read()
#Depressed_words=open("P:/Study/Thesis works/Masum405/Depressed words22.txt","r", encoding = 'utf8').read()


# move this up here
all_words = []
documents = []



for p in Depressed.split('\n'):
    documents.append( (p, "D") )# tag with D
    #print(documents)
    words = word_tokenize(p)
    #print(words)
    filtered_words = remove_stop_words(words)

    #print('filter word = ',filtered_words)
    ## Without D Tag code
    for ii in filtered_words:
          all_words.append(ii)


for p in Not_Depressed.split('\n'):
    documents.append( (p, "ND") )# tag with neg
    words = word_tokenize(p)
    #print(words)
    filtered_words = remove_stop_words(words)

    #print('filter word = ', filtered_words)

    ## Without POS Tag code
    for ii in filtered_words:
          all_words.append(ii)



#Depressed_words= Depressed_words.split('\n')

#all_words.extend(Depressed_words)

#all_words=set(all_words)



save_status = open("P:/Study/Thesis works/Masum405/pickle/documents.pickle","wb")

pickle.dump(documents, save_status)
save_status.close()

# Save a dictionary into a pickle file.
# favorite_color = { "lion": "yellow", "kitty": "red" }
# pickle.dump( favorite_color, open( "save.p", "wb" ) )


all_words = nltk.FreqDist(all_words) # FreqDist that means total frequency count
#print("all_words  = "+ str(all_words))
print("all_words length is = "+ str(len(all_words)))


#for viewing words frequency
#for i in all_words.keys():
#     print(i,all_words[str(i)])



word_features = list(all_words.keys())[: 3467 ]



#print("word_features = "+str(word_features))

save_word_features = open("P:/Study/Thesis works/Masum405/pickle/word_features5k.pickle","wb")
pickle.dump(word_features, save_word_features)
save_word_features.close()



def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features


#print(find_features('ধন্যবাদ আল্লাহকে আমাকে এমন একটা দুঃখে ভরা জীবন দেবার জন্য বিদায় সবাই'))


featuresets = [(find_features(rev), category) for (rev, category) in documents]


#print("featured sets =  "+str(featuresets))
#print(featuresets[1])
print("length before shuffling = "+ str(len(featuresets)))

save_featuresets = open("P:/Study/Thesis works/Masum405/pickle/featuresets.pickle","wb")
pickle.dump(featuresets, save_featuresets)
save_featuresets.close()

random.shuffle(featuresets)


print("length after shuffling = "+ str(len(featuresets)))


training_set = featuresets[: 522 ]
testing_set = featuresets[ 522 :]

# print('training_set = ',str(training_set))# 'নিউজিল্যাণ্ড': False,'দল': True,
# print('testing_set = ',str(testing_set)) #

NB_classifier = nltk.NaiveBayesClassifier.train(training_set)
refsets = collections.defaultdict(set)
testsets = collections.defaultdict(set)

for i, (feats, label) in enumerate(testing_set):
    # print('i = ',str(i),' feats = ',str(feats))
    # print('i = ', str(i), ' label = ', str(label))# label= pos, neg
    refsets[label].add(i)
    # print(str(refsets[label]))
    observed = NB_classifier.classify(feats)
    # print('observed = ',observed)# observed= pos, neg
    testsets[observed].add(i)
    # print( str(testsets[observed]))

# print('refsets = ',str(refsets))
# print('testsets = ',str(testsets))

print("for unigram\n")
print("Original Naive Bayes classifier accuracy:", (nltk.classify.accuracy(NB_classifier, testing_set)) * 100)

# print('confusion matrix')
# print((confusion_matrix(refsets['neg'], testsets['neg']))*100)

print('Depressed precision:', (precision(refsets['D'], testsets['D'])) )
print('D recall:', (recall(refsets['D'], testsets['D'])) )
print('D f_measure:', (f_measure(refsets['D'], testsets['D'])) )
print('Not Depressed precision:', (precision(refsets['ND'], testsets['ND'])))
print('ND recall:', (recall(refsets['ND'], testsets['ND'])) )
print('ND f_measure:', (f_measure(refsets['ND'], testsets['ND'])) )

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
print('D f_measure:', f_measure(refsets['D'], testsets['D']) )
print('ND precision:', precision(refsets['ND'], testsets['ND']) )
print('ND recall:', recall(refsets['ND'], testsets['ND']) )
print('ND f_measure:', f_measure(refsets['ND'], testsets['ND']) )

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)

for i, (feats, label) in enumerate(testing_set):
    refsets[label].add(i)
    observed = LogisticRegression_classifier.classify(feats)
    testsets[observed].add(i)
print("\n")
print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set)) * 100)
print('D precision:', (precision(refsets['D'], testsets['D']))  )
print('D recall:', (recall(refsets['D'], testsets['D']))  )
print('D f_measure:', (f_measure(refsets['D'], testsets['D'])) )
print('ND precision:', (precision(refsets['ND'], testsets['ND'])) )
print('ND recall:', (recall(refsets['ND'], testsets['ND'])) )
print('ND f_measure:', (f_measure(refsets['ND'], testsets['ND'])) )


LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)

for i, (feats, label) in enumerate(testing_set):
    refsets[label].add(i)
    observed = LinearSVC_classifier.classify(feats)
    testsets[observed].add(i)
print("\n")
print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set)) * 100)
print('D precision:', (precision(refsets['D'], testsets['D'])) )
print('D recall:', (recall(refsets['D'], testsets['D'])) )
print('D f_measure:', (f_measure(refsets['D'], testsets['D'])) )
print('ND precision:', (precision(refsets['ND'], testsets['ND'])) )
print('ND recall:', (recall(refsets['ND'], testsets['ND'])) )
print('ND f_measure:', (f_measure(refsets['ND'], testsets['ND'])) )

NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)

for i, (feats, label) in enumerate(testing_set):
    refsets[label].add(i)
    observed = NuSVC_classifier.classify(feats)
    testsets[observed].add(i)
print("\n")
print("NuSVC_classifier accuracy percent:", (nltk.classify.accuracy(NuSVC_classifier, testing_set)) * 100)
print('D precision:', (precision(refsets['D'], testsets['D'])) )
print('D recall:', (recall(refsets['D'], testsets['D'])) )
print('D f_measure:', (f_measure(refsets['D'], testsets['D'])) )
print('ND precision:', (precision(refsets['ND'], testsets['ND'])) )
print('ND recall:', (recall(refsets['ND'], testsets['ND'])) )
print('ND f_measure:', (f_measure(refsets['ND'], testsets['ND'])))

BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)

for i, (feats, label) in enumerate(testing_set):
    refsets[label].add(i)
    observed = BernoulliNB_classifier.classify(feats)
    testsets[observed].add(i)
print("\n")
print("BernoulliNB_classifier accuracy percent:", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set)) * 100)
print('D precision:', (precision(refsets['D'], testsets['D'])) )
print('D recall:', (recall(refsets['D'], testsets['D'])) )
print('D f_measure:', (f_measure(refsets['D'], testsets['D'])) )
print('ND precision:', (precision(refsets['ND'], testsets['ND'])) )
print('ND recall:', (recall(refsets['ND'], testsets['ND'])) )
print('ND f_measure:', (f_measure(refsets['ND'], testsets['ND'])) )

SGDC_classifier = SklearnClassifier(SGDClassifier())
SGDC_classifier.train(training_set)

for i, (feats, label) in enumerate(testing_set):
    refsets[label].add(i)
    observed = SGDC_classifier.classify(feats)
    testsets[observed].add(i)
print("\n")
print("SGDC_classifier accuracy percent:", (nltk.classify.accuracy(SGDC_classifier, testing_set)) * 100)
print('D precision:', (precision(refsets['D'], testsets['D'])) )
print('D recall:', (recall(refsets['D'], testsets['D'])) )
print('D f_measure:', (f_measure(refsets['D'], testsets['D'])) )
print('ND precision:', (precision(refsets['ND'], testsets['ND'])) )
print('ND recall:', (recall(refsets['ND'], testsets['ND'])) )
print('ND f_measure:', (f_measure(refsets['ND'], testsets['ND'])) )

print('\n\nIs it Ok run-2 file??')

