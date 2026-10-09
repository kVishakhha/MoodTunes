import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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

emotion_labels = [
    "sadness",
    "joy",
    "love",
    "anger",
    "fear",
    "surprise"
]

train_df = pd.read_csv("data/training.csv")
test_df = pd.read_csv("data/test.csv")

train_df["emotion"] = train_df["label"].map(label_map)
test_df["emotion"] = test_df["label"].map(label_map)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):

    text = text.lower()

    text = re.sub(
        r"http\S+|www\S+|https\S+",
        "",
        text
    )

    text = re.sub(
        r"[^a-zA-Z\s]",
        "",
        text
    )

    words = text.split()

    words = [
        word
        for word in words
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

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=1000
    ),

    "Naive Bayes": MultinomialNB(),

    "SVM": LinearSVC()
}

for name, model in models.items():

    print(f"\nGenerating confusion matrix for {name}...")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=emotion_labels
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=emotion_labels
    )

    display.plot(
        xticks_rotation=45
    )

    plt.title(f"{name} - Confusion Matrix")

    plt.tight_layout()

    filename = name.lower().replace(" ", "_")

    plt.savefig(
        f"data/{filename}_confusion_matrix.png",
        dpi=300
    )

    plt.show()

    plt.close()

    print(
        f"Saved: data/{filename}_confusion_matrix.png"
    )


print("\nAll confusion matrices generated successfully!")