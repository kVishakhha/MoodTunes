# MoodTunes — Emotion-Aware Music Recommendation

MoodTunes is a small NLP application that predicts the emotion expressed in a short English text and recommends songs from a prepared catalogue for the corresponding mood. It combines text preprocessing, TF–IDF features, a linear Support Vector Machine (SVM), and a Streamlit interface.

## How it works

1. **Input:** The user writes a short description of how they feel.
2. **Preprocessing:** Text is lowercased, contractions and URLs are normalized/removed, non-letter characters are cleaned, English stop words are filtered (while retaining negation words), and remaining words are lemmatized.
3. **Feature extraction:** TfidfVectorizer creates word and n-gram features (1–3 grams, up to 10,000 features).
4. **Emotion classification:** A LinearSVC model predicts one of six labels: sadness, joy, love, anger, fear, or surprise.
5. **Mood mapping:** The predicted emotion maps to a music category.
6. **Recommendation:** MoodTunes selects up to five matching songs from the prepared catalogue and displays their title, artist, and language.

## Emotion-to-mood mapping

| Predicted emotion | Recommendation mood |
| --- | --- |
| Sadness | Calm |
| Fear | Calm |
| Joy | Happy |
| Love | Romantic |
| Anger | Angry |
| Surprise | Energetic |

## Tech stack

- **Python** for the application and model pipeline
- **Pandas** for loading the training data and song catalogue
- **NLTK** for stop-word filtering and lemmatization
- **scikit-learn** for TF–IDF features and LinearSVC
- **Streamlit** for the interactive UI

## Run locally

Use Python 3.10 or newer.

1. Clone the repository and open a terminal in the project root.
2. Create and activate a virtual environment:

   ~~~bash
   python -m venv .venv
   ~~~

   Windows PowerShell:

   ~~~powershell
   .\.venv\Scripts\Activate.ps1
   ~~~

   macOS/Linux:

   ~~~bash
   source .venv/bin/activate
   ~~~

3. Install dependencies and download the NLTK resources used by preprocessing:

   ~~~bash
   pip install -r requirements.txt
   python -m nltk.downloader stopwords wordnet omw-1.4
   ~~~

4. Launch the application:

   ~~~bash
   streamlit run src/streamlit_app.py
   ~~~

Open the local URL printed by Streamlit, usually http://localhost:8501.

## Data files

- **data/training.csv** — labelled emotion examples used to fit the classifier.
- **data/songs.csv** — curated song catalogue used after an emotion is predicted.
- **data/processed_training.csv** — a processed dataset artifact retained in the repository history; the current Streamlit app loads data/training.csv directly.

The song catalogue is used for matching recommendations; it is not used to train the emotion classifier.

## Evaluation and limitations

The app currently trains a lightweight TF–IDF + linear SVM classifier and caches the fitted model for the current Streamlit process. No validated accuracy, precision, recall, or F1-score is claimed in this README because a verified evaluation result was not available when it was prepared.

- The input is English text and the classifier predicts one emotion per submission.
- TF–IDF features do not fully capture context, sarcasm, or complex negation.
- Recommendations come from a prepared catalogue, not a live streaming service.
- The result is a model prediction, not a clinical or definitive assessment of a person's emotional state.

## Repository layout

- **src/streamlit_app.py** — Streamlit UI, model invocation, emotion mapping, and song recommendations
- **src/model_pipeline.py** — shared text preprocessing and TF–IDF configuration
- **data/** — training data and song catalogue
- **notebooks/** — exploratory work, if present

## Possible future work

- Add a reproducible held-out evaluation with per-class precision, recall, F1-score, and a confusion matrix.
- Compare the current linear SVM with other baselines using the same split and preprocessing.
- Add validated screenshots of the running application and evaluation outputs.
- Explore transformer-based representations only after establishing a measurable baseline.
