import pandas as pd

songs = pd.read_csv("data/songs.csv")

emotion_to_mood = {
    "joy": "Happy",
    "sadness": "Calm",
    "love": "Romantic",
    "anger": "Energetic",
    "fear": "Soothing",
    "surprise": "Upbeat"
}

def recommend_songs(emotion, number_of_songs=5):

    mood = emotion_to_mood.get(emotion)

    if mood is None:
        return pd.DataFrame()

    matching_songs = songs[
        songs["mood"] == mood
    ]

    recommendations = matching_songs.sample(
        n=min(number_of_songs, len(matching_songs)),
        random_state=42
    )

    return recommendations

test_emotion = "joy"

recommended = recommend_songs(
    test_emotion,
    number_of_songs=5
)


print("\nDetected Emotion:")
print(test_emotion)

print("\nRecommended Mood:")
print(emotion_to_mood[test_emotion])

print("\nRecommended Songs:")

for _, song in recommended.iterrows():

    print(
        f"{song['song']} - "
        f"{song['artist']} "
        f"({song['language']})"
    )