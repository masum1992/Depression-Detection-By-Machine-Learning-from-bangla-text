#Depression Detection Model With Python

import nltk
import random
from nltk.classify.scikitlearn import SklearnClassifier
import pickle
import bangla_pos_tagger
from stop_words_remover import remove_stop_words

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



word_features = list(all_words.keys())[: 3684 ]



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


training_set = featuresets[: 514 ]
testing_set = featuresets[ 514 :]


classifier = nltk.NaiveBayesClassifier.train(training_set)
print("\nOriginal Naive Bayes Algo accuracy percent:", (nltk.classify.accuracy(classifier, testing_set))*100)
print()
classifier.show_most_informative_features(10)

###############
save_classifier = open("P:/Study/Thesis works/Masum405/pickle/originalnaivebayes5k.pickle","wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("\nMNB_classifier accuracy percent:", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)


save_classifier = open("P:/Study/Thesis works/Masum405/pickle/MNB_classifier5k.pickle","wb")
pickle.dump(MNB_classifier, save_classifier)
save_classifier.close()

BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print("\nBernoulliNB_classifier accuracy percent:", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set))*100)

save_classifier = open("P:/Study/Thesis works/Masum405/pickle/BernoulliNB_classifier5k.pickle","wb")
pickle.dump(BernoulliNB_classifier, save_classifier)
save_classifier.close()

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("\nLogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)

save_classifier = open("P:/Study/Thesis works/Masum405/pickle/LogisticRegression_classifier5k.pickle","wb")
pickle.dump(LogisticRegression_classifier, save_classifier)
save_classifier.close()


LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("\nLinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)

save_classifier = open("P:/Study/Thesis works/Masum405/pickle/LinearSVC_classifier5k.pickle","wb")
pickle.dump(LinearSVC_classifier, save_classifier)
save_classifier.close()


NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print("\nNuSVC_classifier accuracy percent:", (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)
print()

save_classifier = open("P:/Study/Thesis works/Masum405/pickle/NuSVC_classifier5k.pickle","wb")
pickle.dump(NuSVC_classifier, save_classifier)
save_classifier.close()

# SGDC_classifier = SklearnClassifier(SGDClassifier())
# SGDC_classifier.train(training_set)
# print("\nSGDClassifier accuracy percent:",nltk.classify.accuracy(SGDC_classifier, testing_set)*100)

# save_classifier = open("E:/ThesisCodeTest/pickle/SGDC_classifier5k.pickle","wb")
# pickle.dump(SGDC_classifier, save_classifier)
# save_classifier.close()

print('\n\n End of result')


