import pandas as pd
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

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

df["clean_text"] = df["text"].apply(preprocess_text)

print("\nOriginal vs Cleaned Text:\n")

for i in range(5):

    print("Original:")
    print(df.iloc[i]["text"])

    print("\nCleaned:")
    print(df.iloc[i]["clean_text"])

    print("\nEmotion:")
    print(df.iloc[i]["emotion"])

    print("-" * 60)

df.to_csv("data/processed_training.csv", index=False)

print("\nProcessed dataset saved successfully!")