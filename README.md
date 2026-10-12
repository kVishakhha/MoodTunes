# MoodTunes

MoodTunes detects the emotion expressed in a short text and recommends songs for a matching mood. It uses text preprocessing, TF-IDF features, and a support vector machine, with a Streamlit interface.

## Run locally

1. Install Python 3.10 or newer.
2. From the project root, create and activate a virtual environment:

   ```bash
   python -m venv venv
   ```

   On Windows PowerShell:

   ```powershell
   .\venv\Scripts\Activate.ps1
   ```

   On macOS or Linux:

   ```bash
   source venv/bin/activate
   ```

3. Install dependencies and the NLTK data used for preprocessing:

   ```bash
   pip install -r requirements.txt
   python -m nltk.downloader stopwords wordnet omw-1.4
   ```

4. Start the app from the project root:

   ```bash
   streamlit run src/streamlit_app.py
   ```

Open the local URL printed by Streamlit, usually `http://localhost:8501`. The app trains its model the first time you submit a non-empty mood description; the cached model is reused until the app process restarts.

## Project layout

- `src/` — application, model, preprocessing, and evaluation scripts
- `data/` — datasets and song catalog
- `notebooks/` — exploratory notebooks
- `models/` — model artifacts, if generated

## Emotion to mood mapping

The app maps sadness and fear to Calm, joy to Happy, love to Romantic, anger to Angry, and surprise to Energetic.
