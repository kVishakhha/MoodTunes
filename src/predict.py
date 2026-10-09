import pandas as pd
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

df = pd.read_csv("data/training.csv")

label_map = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise"
}

df["emotion"] = df["label"].map(label_map)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):

    text = text.lower()

    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    text = re.sub(r"[^a-zA-Z\s]", "", text)

    words = text.split()

    words = [
        word for word in words
        if word not in stop_words
    ]

    words = [
        lemmatizer.lemmatize(word)
        for word in words
    ]

    return " ".join(words)

print("Preprocessing training data...")

X_text = df["text"].apply(preprocess_text)
y = df["emotion"]

print("Creating TF-IDF features...")

vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)

X = vectorizer.fit_transform(X_text)

print("Training SVM model...")

model = LinearSVC()
model.fit(X, y)

print("Model trained successfully!")

user_text = input("\nEnter how you are feeling: ")

clean_text = preprocess_text(user_text)

text_vector = vectorizer.transform([clean_text])

predicted_emotion = model.predict(text_vector)[0]

print("\n-----------------------------")
print("MoodTunes Prediction")
print("-----------------------------")

print("\nYour text:")
print(user_text)

print("\nDetected Emotion:")
print(predicted_emotion)