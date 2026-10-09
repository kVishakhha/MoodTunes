"""Shared text preprocessing and TF-IDF settings for MoodTunes."""
import re

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer


STOP_WORDS = set(stopwords.words("english")) - {"no", "not", "never"}
LEMMATIZER = WordNetLemmatizer()
CONTRACTIONS = {
    "don't": "do not", "doesn't": "does not", "didn't": "did not",
    "can't": "can not", "couldn't": "could not", "wouldn't": "would not",
    "shouldn't": "should not", "won't": "will not", "isn't": "is not",
    "aren't": "are not", "wasn't": "was not", "weren't": "were not",
    "haven't": "have not", "hasn't": "has not", "hadn't": "had not",
}
# The dataset also contains common contractions without apostrophes (for example,
# "dont" and "didnt"). Normalize those to the same tokens as user-entered text.
CONTRACTIONS.update({
    "dont": "do not", "doesnt": "does not", "didnt": "did not",
    "cant": "can not", "couldnt": "could not", "wouldnt": "would not",
    "shouldnt": "should not", "wont": "will not", "isnt": "is not",
    "arent": "are not", "wasnt": "was not", "werent": "were not",
    "havent": "have not", "hasnt": "has not", "hadnt": "had not",
})
CONTRACTION_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(item) for item in sorted(CONTRACTIONS, key=len, reverse=True)) + r")\b"
)


def preprocess_text(text):
    text = str(text).lower()
    text = CONTRACTION_PATTERN.sub(lambda match: CONTRACTIONS[match.group(0)], text)
    text = re.sub(r"(?:https?://|www\.)\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    words = (word for word in text.split() if word not in STOP_WORDS)
    return " ".join(LEMMATIZER.lemmatize(word) for word in words)


def make_vectorizer():
    return TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 3),
        sublinear_tf=True,
    )
