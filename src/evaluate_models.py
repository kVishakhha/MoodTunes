import pandas as pd
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

label_map = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise"
}

train_df = pd.read_csv("data/training.csv")
test_df = pd.read_csv("data/test.csv")

train_df["emotion"] = train_df["label"].map(label_map)
test_df["emotion"] = test_df["label"].map(label_map)

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

train_text = train_df["text"].apply(preprocess_text)

print("Preprocessing test data...")

test_text = test_df["text"].apply(preprocess_text)

vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)

X_train = vectorizer.fit_transform(train_text)

X_test = vectorizer.transform(test_text)

y_train = train_df["emotion"]
y_test = test_df["emotion"]


print("\nTF-IDF completed!")
print("Training matrix:", X_train.shape)
print("Testing matrix:", X_test.shape)

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=1000
    ),

    "Naive Bayes": MultinomialNB(),

    "SVM": LinearSVC()
}

results = []


for name, model in models.items():

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        average="weighted"
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="weighted"
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted"
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1
    })

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-Score : {f1:.4f}")

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            y_pred
        )
    )

results_df = pd.DataFrame(results)

print("\n\n")
print("=" * 70)
print("FINAL MODEL COMPARISON")
print("=" * 70)

print(results_df.to_string(index=False))

results_df.to_csv(
    "data/model_comparison.csv",
    index=False
)

print("\nModel comparison saved to:")
print("data/model_comparison.csv")