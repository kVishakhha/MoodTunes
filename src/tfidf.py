import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv("data/processed_training.csv")

texts = df["clean_text"]

vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)

X = vectorizer.fit_transform(texts)

y = df["emotion"]

print("TF-IDF conversion completed!")

print("\nNumber of text samples:")
print(X.shape[0])

print("\nNumber of TF-IDF features:")
print(X.shape[1])

print("\nTF-IDF matrix shape:")
print(X.shape)

print("\nEmotion labels:")
print(y.head().to_string(index=False))

print("\nFirst text:")
print(texts.iloc[0])

print("\nNon-zero TF-IDF values for first text:")

first_vector = X[0].toarray()[0]

print(first_vector[first_vector > 0])