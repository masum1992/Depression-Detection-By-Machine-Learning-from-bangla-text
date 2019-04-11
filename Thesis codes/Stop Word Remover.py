from nltk.tokenize import word_tokenize


def remove_stop_words(words):
    stop_words = word_tokenize(open("P:/Study/Thesis works/Masum405/stopwords.txt", "r", encoding="utf-8").read())
    filtered_sentence = []

    for w in words:
        if w not in stop_words:
            filtered_sentence.append(w)

    return filtered_sentence