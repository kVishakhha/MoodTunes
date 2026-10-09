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

stop_words = stop_words - {"no", "not", "never"}

lemmatizer = WordNetLemmatizer()

contractions = {
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "can't": "can not",
    "couldn't": "could not",
    "wouldn't": "would not",
    "shouldn't": "should not",
    "won't": "will not",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "haven't": "have not",
    "hasn't": "has not",
    "hadn't": "had not",
    "don't": "do not"
}

label_map = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise"
}

training_data["emotion"] = training_data["label"].map(label_map)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):

    text = text.lower()

    for contraction, replacement in contractions.items():
        text = text.replace(
            contraction,
            replacement
        )

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

print("Preparing MoodTunes...")

training_text = training_data["text"].apply(
    preprocess_text
)

y = training_data["emotion"]

vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 3),
    sublinear_tf=True
)

X = vectorizer.fit_transform(
    training_text
)

model = LinearSVC(
    class_weight="balanced"
)

model.fit(X, y)

print("Model ready!")

emotion_to_mood = {
    "joy": "Happy",
    "sadness": "Calm",
    "love": "Romantic",
    "anger": "Energetic",
    "fear": "Soothing",
    "surprise": "Upbeat"
}

def detect_emotion(text):

    clean_text = preprocess_text(text)

    text_vector = vectorizer.transform(
        [clean_text]
    )

    emotion = model.predict(text_vector)[0]

    return emotion

def recommend_songs(
    emotion,
    number_of_songs=5
):

    mood = emotion_to_mood.get(emotion)

    if mood is None:
        return pd.DataFrame()

    matching_songs = songs[
        songs["mood"] == mood
    ]

    recommendations = matching_songs.sample(
        n=min(
            number_of_songs,
            len(matching_songs)
        ),
        random_state=42
    )

    return recommendations

print("\n================================")
print("          MOODTUNES")
print("================================")

user_text = input(
    "\nHow are you feeling today?\n> "
)


emotion = detect_emotion(user_text)

mood = emotion_to_mood.get(emotion)

recommendations = recommend_songs(
    emotion,
    number_of_songs=5
)

print("\n================================")
print("         MOODTUNES RESULT")
print("================================")

print("\nYour text:")
print(user_text)

print("\nDetected Emotion:")
print(emotion)

print("\nRecommended Mood:")
print(mood)

print("\nRecommended Songs:")
print("--------------------------------")

for index, song in recommendations.iterrows():

    print(
        f"{song['song']} - "
        f"{song['artist']} "
        f"({song['language']})"
    )

print("\n================================")
print("       Enjoy your music! 🎵")
print("================================")