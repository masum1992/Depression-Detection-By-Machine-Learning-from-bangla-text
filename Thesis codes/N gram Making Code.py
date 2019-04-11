#For n-grams
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
import pandas as pd
from stop_words_remover import remove_stop_words

from sklearn.feature_extraction.text import TfidfVectorizer



# move this up here
all_words = []
documents = []



#N-Grams making function

# def get_ngrams(tokens, n ):
#     n_grams = ngrams(tokens, n)
#
#     return [ ' '.join(grams) for grams in n_grams]
#


def get_ngrams(text, n ):
    n_grams = ngrams(word_tokenize(text), n)
    return [ ' '.join(grams) for grams in n_grams]




myvocabulary = ['tim tam', 'jam', 'fresh milk', 'chocolates', 'biscuit pudding']
corpus = {1: "making chocolates biscuit pudding easy first get your favourite biscuit chocolates", 2: "tim tam drink new recipe that yummy and tasty more thicker than typical milkshake that uses normal chocolates", 3: "making chocolates drink different way using fresh milk egg"}



tfidf = TfidfVectorizer(vocabulary = myvocabulary, stop_words = 'english', ngram_range=(1,2))
tfs = tfidf.fit_transform(corpus.values())


feature_names = tfidf.get_feature_names()

corpus_index = [n for n in corpus]
rows, cols = tfs.nonzero()
for row, col in zip(rows, cols):
    print((feature_names[col], corpus_index[row]), tfs[row, col])


import pandas as pd
df = pd.DataFrame(tfs.T.todense(), index=feature_names, columns=corpus_index)
print(df)
