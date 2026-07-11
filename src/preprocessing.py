import string
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize


nltk.download("punkt")
nltk.download("stopwords")


stemmer = PorterStemmer()


def transform_text(text):

    # lowercase
    text = text.lower()

    # tokenize
    words = word_tokenize(text)

    # remove special characters
    words = [
        word for word in words
        if word.isalnum()
    ]

    # remove stopwords
    words = [
        word for word in words
        if word not in stopwords.words("english")
    ]

    # stemming
    words = [
        stemmer.stem(word)
        for word in words
    ]

    return " ".join(words)