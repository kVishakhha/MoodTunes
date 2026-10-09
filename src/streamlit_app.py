import streamlit as st
import pandas as pd
from sklearn.svm import LinearSVC

from model_pipeline import make_vectorizer, preprocess_text

st.set_page_config(
    page_title="MoodTunes",
    page_icon="🎵",
    layout="centered"
)

@st.cache_data
def load_data():

    training_data = pd.read_csv(
        "data/training.csv"
    )

    songs = pd.read_csv(
        "data/songs.csv"
    )

    return training_data, songs


training_data, songs = load_data()

label_map = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise"
}

training_data["emotion"] = training_data[
    "label"
].map(label_map)

@st.cache_resource
def train_model():

    training_text = training_data[
        "text"
    ].apply(preprocess_text)

    y = training_data[
        "emotion"
    ]

    
    vectorizer = make_vectorizer()

    X = vectorizer.fit_transform(
        training_text
    )

    
    model = LinearSVC(
        C=0.1,
        class_weight="balanced"
    )

    model.fit(
        X,
        y
    )

    return vectorizer, model


vectorizer, model = train_model()

emotion_to_mood = {

    "joy": "Happy",

    "sadness": "Calm",

    "love": "Romantic",

    "anger": "Energetic",

    "fear": "Soothing",

    "surprise": "Upbeat"
}

emotion_emoji = {

    "joy": "😊",

    "sadness": "😢",

    "love": "❤️",

    "anger": "😡",

    "fear": "😨",

    "surprise": "😲"
}

def detect_emotion(text):

    clean_text = preprocess_text(
        text
    )

    text_vector = vectorizer.transform(
        [clean_text]
    )

    emotion = model.predict(
        text_vector
    )[0]

    return emotion


def recommend_songs(
    emotion,
    number_of_songs=5
):

    mood = emotion_to_mood.get(
        emotion
    )

    if mood is None:
        return pd.DataFrame()

    matching_songs = songs[
        songs["mood"] == mood
    ]

    if len(matching_songs) == 0:
        return pd.DataFrame()

    recommendations = matching_songs.sample(
        n=min(
            number_of_songs,
            len(matching_songs)
        ),
        random_state=42
    )

    return recommendations


st.title("🎵 MoodTunes")

st.subheader(
    "Music that matches your mood"
)

st.write(
    "Tell us how you're feeling, and MoodTunes "
    "will detect your emotion and recommend songs "
    "that match your mood."
)


user_text = st.text_area(
    "How are you feeling today?",
    placeholder=(
        "Example: I am feeling happy and "
        "excited today..."
    ),
    height=150
)


if st.button(
    "🎧 Detect My Mood",
    use_container_width=True
):

    if not user_text.strip():

        st.warning(
            "Please enter something about "
            "how you're feeling."
        )

    else:

        emotion = detect_emotion(
            user_text
        )

        mood = emotion_to_mood.get(
            emotion
        )

        recommendations = recommend_songs(
            emotion,
            number_of_songs=5
        )

        st.success(
            f"Detected Emotion: "
            f"{emotion_emoji.get(emotion, '')} "
            f"{emotion.title()}"
        )

        st.info(
            f"Recommended Mood: 🎵 {mood}"
        )

        st.subheader(
            "🎶 Recommended Songs"
        )

        if recommendations.empty:

            st.warning(
                "No songs were found for "
                "this mood."
            )

        else:

            for _, song in recommendations.iterrows():

                st.markdown(
                    f"**{song['song']}** — "
                    f"{song['artist']} "
                    f"({song['language']})"
                )

st.markdown("---")

st.caption(
    "MoodTunes uses NLP, TF-IDF and SVM "
    "to detect emotions from text."
)
