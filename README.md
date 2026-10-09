# MoodTunes

MoodTunes detects the emotion expressed in a short text and recommends songs for a matching mood. It uses text preprocessing, TF-IDF features, and a support vector machine, with a Streamlit interface.

## Run locally

1. Install Python 3.10 or newer.
2. Create and activate a virtual environment.
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Start the app from the project root:

   ```bash
   streamlit run src/streamlit_app.py
   ```

The app reads its training and song data from `data/training.csv` and `data/songs.csv`. NLTK corpora used by preprocessing may need to be downloaded once (`stopwords`, `wordnet`, and `omw-1.4`).

## Project layout

- `src/` — application, model, preprocessing, and evaluation scripts
- `data/` — datasets and song catalog
- `notebooks/` — exploratory notebooks
- `models/` — model artifacts, if generated

## Emotion to mood mapping

The app maps sadness to Calm, joy to Happy, love to Romantic, anger to Energetic, fear to Soothing, and surprise to Upbeat.
