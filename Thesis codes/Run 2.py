
import nltk
import random
import pickle

import nltk.classify.util
from nltk.metrics.scores import precision, recall, f_measure
import collections

# import Stemmer
import bangla_pos_tagger
from stop_words_remover import remove_stop_words

from nltk.classify import NaiveBayesClassifier, MaxentClassifier
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

from sklearn.metrics import confusion_matrix

from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize

######### import for bigram########
import itertools
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures

btagger = bangla_pos_tagger.BanglaTagger()

short_pos = open("P:/Study/Thesis works/Masum405/testfile/Depressed22.txt","r", encoding = 'utf8').read()
short_neg = open("P:/Study/Thesis works/Masum405/testfile/Not Depressed22.txt","r", encoding = 'utf8').read()

# move this up here
all_words = []
documents = []

# allowed_word_types = ["J","R","V"]
allowed_word_types = ["JJ", "NN","NC", "PP", "NNP"]

for p in short_pos.split('\n'):  # Notice here
    documents.append((p, "pos"))
    # p = Stemmer.stem(p)
    words = word_tokenize(p)
    filtered_words = remove_stop_words(words)
    ##    documents.append( (filtered_words, "pos") )

    ## Without POS Tag code
    for ii in filtered_words:
        #if ii != '\ufeff':
          # print("ii = " + str(ii))
        all_words.append(ii)

    # print("filtered words = "+str(filtered_words))

    # pos = btagger.pos_tag(filtered_words)
    # for w in pos:
    #     if w[1] in allowed_word_types:
    #         all_words.append(w[0])

for p in short_neg.split('\n'):  # NOtice here
    documents.append((p, "neg"))
    # p = Stemmer.stem(p)
    words = word_tokenize(p)
    filtered_words = remove_stop_words(words)
    ##    documents.append( (filtered_words, "neg") )

    ## Without POS Tag code
    for ii in filtered_words:
        #if ii != '\ufeff':
         all_words.append(ii)

    # pos = btagger.pos_tag(filtered_words)
    # for w in pos:
    #     if w[1] in allowed_word_types:
    #         all_words.append(w[0])

all_words = nltk.FreqDist(all_words)

print('all_words len = ',str(len(all_words)))

word_features = list(all_words.keys())[:3684]  # Notice here


#######Unigram feature extraction##########

def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features


featuresets = [(find_features(rev), category) for (rev, category) in documents]

random.shuffle(featuresets)
print("featuresets length = ", len(featuresets))
# print('featuresets = ',str(featuresets))# 'কাল': False, 'জিতেই': False, 'ব্যপারে': False

training_set = featuresets[:514]  ######Notice here
testing_set = featuresets[514:]  ######Notice here

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

print('pos precision:', (precision(refsets['pos'], testsets['pos'])) * 100)
print('pos recall:', (recall(refsets['pos'], testsets['pos'])) * 100)
print('pos f_measure:', (f_measure(refsets['pos'], testsets['pos'])) * 100)
print('neg precision:', (precision(refsets['neg'], testsets['neg'])) * 100)
print('neg recall:', (recall(refsets['neg'], testsets['neg'])) * 100)
print('neg f_measure:', (f_measure(refsets['neg'], testsets['neg'])) * 100)

NB_classifier.show_most_informative_features(10)

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)

for i, (feats, label) in enumerate(testing_set):
    refsets[label].add(i)
    observed = MNB_classifier.classify(feats)
    testsets[observed].add(i)
print("\n")
print("MNB_classifier accuracy percent:", (nltk.classify.accuracy(MNB_classifier, testing_set)) * 100)
print('pos precision:', precision(refsets['pos'], testsets['pos']))
print('pos recall:', recall(refsets['pos'], testsets['pos']))
print('pos f_measure:', f_measure(refsets['pos'], testsets['pos']))
print('neg precision:', precision(refsets['neg'], testsets['neg']))
print('neg recall:', recall(refsets['neg'], testsets['neg']))
print('neg f_measure:', f_measure(refsets['neg'], testsets['neg']))

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)

for i, (feats, label) in enumerate(testing_set):
    refsets[label].add(i)
    observed = LogisticRegression_classifier.classify(feats)
    testsets[observed].add(i)
print("\n")
print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set)) * 100)
print('pos precision:', (precision(refsets['pos'], testsets['pos'])) * 100)
print('pos recall:', (recall(refsets['pos'], testsets['pos'])) * 100)
print('pos f_measure:', (f_measure(refsets['pos'], testsets['pos'])) * 100)
print('neg precision:', (precision(refsets['neg'], testsets['neg'])) * 100)
print('neg recall:', (recall(refsets['neg'], testsets['neg'])) * 100)
print('neg f_measure:', (f_measure(refsets['neg'], testsets['neg'])) * 100)


LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)

for i, (feats, label) in enumerate(testing_set):
    refsets[label].add(i)
    observed = LinearSVC_classifier.classify(feats)
    testsets[observed].add(i)
print("\n")
print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set)) * 100)
print('pos precision:', (precision(refsets['pos'], testsets['pos'])) * 100)
print('pos recall:', (recall(refsets['pos'], testsets['pos'])) * 100)
print('pos f_measure:', (f_measure(refsets['pos'], testsets['pos'])) * 100)
print('neg precision:', (precision(refsets['neg'], testsets['neg'])) * 100)
print('neg recall:', (recall(refsets['neg'], testsets['neg'])) * 100)
print('neg f_measure:', (f_measure(refsets['neg'], testsets['neg'])) * 100)

NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)

for i, (feats, label) in enumerate(testing_set):
    refsets[label].add(i)
    observed = NuSVC_classifier.classify(feats)
    testsets[observed].add(i)
print("\n")
print("NuSVC_classifier accuracy percent:", (nltk.classify.accuracy(NuSVC_classifier, testing_set)) * 100)
print('pos precision:', (precision(refsets['pos'], testsets['pos'])) * 100)
print('pos recall:', (recall(refsets['pos'], testsets['pos'])) * 100)
print('pos f_measure:', (f_measure(refsets['pos'], testsets['pos'])) * 100)
print('neg precision:', (precision(refsets['neg'], testsets['neg'])) * 100)
print('neg recall:', (recall(refsets['neg'], testsets['neg'])) * 100)
print('neg f_measure:', (f_measure(refsets['neg'], testsets['neg'])) * 100)

BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)

for i, (feats, label) in enumerate(testing_set):
    refsets[label].add(i)
    observed = BernoulliNB_classifier.classify(feats)
    testsets[observed].add(i)
print("\n")
print("BernoulliNB_classifier accuracy percent:", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set)) * 100)
print('pos precision:', (precision(refsets['pos'], testsets['pos'])) * 100)
print('pos recall:', (recall(refsets['pos'], testsets['pos'])) * 100)
print('pos f_measure:', (f_measure(refsets['pos'], testsets['pos'])) * 100)
print('neg precision:', (precision(refsets['neg'], testsets['neg'])) * 100)
print('neg recall:', (recall(refsets['neg'], testsets['neg'])) * 100)
print('neg f_measure:', (f_measure(refsets['neg'], testsets['neg'])) * 100)

SGDC_classifier = SklearnClassifier(SGDClassifier())
SGDC_classifier.train(training_set)

for i, (feats, label) in enumerate(testing_set):
    refsets[label].add(i)
    observed = SGDC_classifier.classify(feats)
    testsets[observed].add(i)
print("\n")
print("SGDC_classifier accuracy percent:", (nltk.classify.accuracy(SGDC_classifier, testing_set)) * 100)
print('pos precision:', (precision(refsets['pos'], testsets['pos'])) * 100)
print('pos recall:', (recall(refsets['pos'], testsets['pos'])) * 100)
print('pos f_measure:', (f_measure(refsets['pos'], testsets['pos'])) * 100)
print('neg precision:', (precision(refsets['neg'], testsets['neg'])) * 100)
print('neg recall:', (recall(refsets['neg'], testsets['neg'])) * 100)
print('neg f_measure:', (f_measure(refsets['neg'], testsets['neg'])) * 100)

print('\n\nIs it Ok run-2 file??')