import streamlit as st
import pandas as pd
from sklearn.svm import LinearSVC

from model_pipeline import make_vectorizer, preprocess_text

st.set_page_config(
    page_title="MoodTunes",
    page_icon="🎵",
    layout="centered",
    initial_sidebar_state="collapsed",
)


st.markdown(
    """
    <style>
    .block-container {
        max-width: 920px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    .mood-hero {
        position: relative;
        overflow: hidden;
        padding: 2.5rem 2.6rem;
        margin: 0.5rem 0 1.5rem;
        border: 1px solid #d8eadb;
        border-radius: 28px;
        background: linear-gradient(135deg, #f1faf2 0%, #ffffff 68%, #e6f5e8 100%);
    }
    .mood-hero::after {
        position: absolute;
        top: -74px;
        right: -42px;
        width: 230px;
        height: 230px;
        border: 32px solid rgba(39, 125, 67, 0.08);
        border-radius: 50%;
        content: "";
    }
    .mood-eyebrow {
        color: #277d43;
        font-size: 0.76rem;
        font-weight: 750;
        letter-spacing: 0.15em;
        text-transform: uppercase;
    }
    .mood-hero h1 {
        max-width: 620px;
        margin: 0.7rem 0 0.6rem;
        color: #173b25;
        font-size: clamp(2.1rem, 6vw, 3.5rem);
        line-height: 1.08;
        letter-spacing: -0.045em;
    }
    .mood-hero p {
        max-width: 560px;
        margin: 0;
        color: #52665a;
        font-size: 1.05rem;
        line-height: 1.7;
    }
    .mood-prompt {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        margin-top: 1.3rem;
        padding: 0.55rem 0.85rem;
        border: 1px solid #d8eadb;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.8);
        color: #277d43;
        font-size: 0.82rem;
        font-weight: 650;
    }
    div[data-testid="stTextArea"] textarea {
        border-radius: 14px;
    }
    div[data-testid="stFormSubmitButton"] button {
        min-height: 3rem;
        border-radius: 12px;
        font-weight: 700;
    }
    .mood-footer {
        padding-top: 1.2rem;
        border-top: 1px solid #e5eee6;
        color: #718078;
        font-size: 0.82rem;
        text-align: center;
    }
    @media (max-width: 640px) {
        .block-container {
            padding: 1rem 1rem 2rem;
        }
        .mood-hero {
            padding: 1.8rem 1.4rem;
            border-radius: 22px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
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


emotion_to_mood = {

    "joy": "Happy",

    "sadness": "Calm",

    "love": "Romantic",

    "anger": "Angry",

    "fear": "Calm",

    "surprise": "Energetic"
}

emotion_emoji = {

    "joy": "😊",

    "sadness": "😢",

    "love": "❤️",

    "anger": "😡",

    "fear": "😨",

    "surprise": "😲"
}

def detect_emotion(text, vectorizer, model):
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


st.markdown(
    """
    <div class="mood-hero">
      <div class="mood-eyebrow">Your personal soundtrack</div>
      <h1>Find the music in your mood.</h1>
      <p>Share what is on your mind. MoodTunes will find the feeling behind your words and pair it with songs to match.</p>
      <div class="mood-prompt">✳ &nbsp; A little more you, in every playlist</div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.container(border=True):
    st.subheader("What’s your mood today?")
    st.caption("A sentence or two is all we need to find your sound.")

    with st.form("mood_form"):
        user_text = st.text_area(
            "Tell us how you’re feeling",
            placeholder="For example: I had a long day, but I’m proud of what I got done.",
            height=135,
            max_chars=1000,
        )
        submitted = st.form_submit_button(
            "Find my soundtrack",
            type="primary",
            icon=":material/graphic_eq:",
            width="stretch",
        )

if submitted:
    if not user_text.strip():
        st.warning("Write a little about how you’re feeling, then try again.")
    else:
        with st.spinner("Finding the feeling in your words..."):
            vectorizer, model = train_model()
            emotion = detect_emotion(user_text, vectorizer, model)
            mood = emotion_to_mood.get(emotion)
            recommendations = recommend_songs(emotion, number_of_songs=5)

        st.write("")
        st.markdown("### Your mood, your music")
        emotion_col, mood_col = st.columns(2)
        with emotion_col:
            with st.container(border=True):
                st.caption("WE HEARD")
                st.subheader(f"{emotion_emoji.get(emotion, '')} {emotion.title()}")
        with mood_col:
            with st.container(border=True):
                st.caption("YOUR SOUND")
                st.subheader(f"🎵 {mood}")

        st.write("")
        st.markdown("### Made for this moment")
        if recommendations.empty:
            st.info("We couldn’t find songs for this mood just yet.")
        else:
            for _, song in recommendations.iterrows():
                with st.container(border=True):
                    st.markdown(f"**{song['song']}**")
                    st.caption(f"{song['artist']}  ·  {song['language']}")

st.markdown(
    '<div class="mood-footer">MoodTunes · A soundtrack for wherever you are.</div>',
    unsafe_allow_html=True,
)
