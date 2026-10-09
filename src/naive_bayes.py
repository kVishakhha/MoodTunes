import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("data/processed_training.csv")

X_text = df["clean_text"]
y = df["emotion"]

vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)

X = vectorizer.fit_transform(X_text)

model = MultinomialNB()

model.fit(X, y)

y_pred = model.predict(X)

accuracy = accuracy_score(y, y_pred)

print("\nNaive Bayes trained successfully!")

print("\nTraining Accuracy:")
print(f"{accuracy * 100:.2f}%")

print("\nClassification Report:")

print(
    classification_report(
        y,
        y_pred
    )
)