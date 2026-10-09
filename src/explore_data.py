import pandas as pd

df = pd.read_csv("data/training.csv")

print("Dataset shape:")
print(df.shape)

print("\nColumns:")
print(df.columns)

print("\nEmotion labels:")
print(df["label"].value_counts().sort_index())

label_map = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise"
}

df["emotion"] = df["label"].map(label_map)

print("\nEmotion distribution:")
print(df["emotion"].value_counts())

print("\nSample texts:")

for i in range(5):
    print(f"\n{i + 1}. {df.iloc[i]['text']}")
    print("Label:", df.iloc[i]["label"])
    print("Emotion:", df.iloc[i]["emotion"])