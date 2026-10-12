# MoodTunes 🎵

MoodTunes is a Streamlit application that predicts the emotion expressed in a short text and recommends songs for a matching mood. The app uses a TF-IDF text representation and a linear Support Vector Machine (SVM).

## How it works

1. **Input:** The user describes how they feel.
2. **Preprocessing:** Text is lowercased, contractions are normalized, URLs and non-letter characters are removed, and English stop words are filtered (while preserving negation). Remaining words are lemmatized.
3. **Feature extraction:** TF-IDF converts the cleaned text into unigram, bigram, and trigram features.
4. **Emotion classification:** A linear SVM predicts one of six emotions.
5. **Mood mapping and recommendation:** The predicted emotion maps to a listening mood, and the app samples matching tracks from the song catalog.

| Predicted emotion | Recommendation mood |
|---|---|
| Joy | Happy |
| Sadness | Calm |
| Love | Romantic |
| Anger | Angry |
| Fear | Calm |
| Surprise | Energetic |

## Model evaluation

The repository includes recorded comparisons for Logistic Regression, Multinomial Naive Bayes, and SVM in `data/model_comparison.csv`. These are the saved results from the evaluation script's configuration (TF-IDF with up to 5,000 features and 1–2-grams).

| Model | Accuracy | Weighted precision | Weighted recall | Weighted F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 88.60% | 88.66% | 88.60% | 88.23% |
| Multinomial Naive Bayes | 80.35% | 83.30% | 80.35% | 77.82% |
| Linear SVM | 90.20% | 90.13% | 90.20% | 90.15% |

The saved SVM tuning results in `data/svm_tuning_results.csv` report the best accuracy in that experiment at **C = 2.0** (91.00%; macro F1 88.00%). These experiments use a different feature configuration from the current Streamlit app, which uses up to 10,000 TF-IDF features, 1–3-grams, `C=0.1`, and `class_weight="balanced"`. Treat the tables as recorded experiment results, not as a claim that the current app configuration has been independently re-evaluated with the same settings.

### Reproduce the model comparison

From the repository root, install the dependencies and NLTK resources, then run:

```bash
pip install -r requirements.txt
python -m nltk.downloader stopwords wordnet omw-1.4
python src/evaluate_models.py
```

The evaluation script reads `data/training.csv` and `data/test.csv` and rewrites `data/model_comparison.csv`. It prints a classification report for each model. Run `python src/confusion_matrix.py` to generate confusion-matrix plots.

## Dataset

The repository contains the training, validation, and test splits under `data/`, with six emotion labels: sadness, joy, love, anger, fear, and surprise. The app's song catalog is stored separately in `data/songs.csv`.

**Dataset provenance:** The current repository documentation does not identify the original source, version, or license of the emotion dataset. Add the original dataset citation and license here before redistributing the data or presenting the project as reproducible research. The preprocessing script can generate `data/processed_training.csv`, but this redundant output is ignored by Git; the app itself reads `data/training.csv`.

## Run locally

**Requirements:** Python 3.10 or newer.

1. Clone the repository and enter it:

   ```bash
   git clone https://github.com/kVishakhha/MoodTunes.git
   cd MoodTunes
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   ```

   Windows PowerShell:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

   macOS / Linux:

   ```bash
   source .venv/bin/activate
   ```

3. Install dependencies and NLTK data:

   ```bash
   pip install -r requirements.txt
   python -m nltk.downloader stopwords wordnet omw-1.4
   ```

4. Start the application:

   ```bash
   streamlit run src/streamlit_app.py
   ```

Open the local URL shown by Streamlit, usually http://localhost:8501. The app trains and caches its model when first needed.

## Repository structure

- `src/streamlit_app.py` — Streamlit user interface and recommendation flow
- `src/model_pipeline.py` — shared preprocessing and TF-IDF configuration
- `src/evaluate_models.py` — baseline model comparison
- `src/confusion_matrix.py` — confusion-matrix generation
- `src/` — model, prediction, preprocessing, and recommendation modules
- `data/training.csv`, `data/validation.csv`, `data/test.csv` — dataset splits
- `data/songs.csv` — song catalog
- `data/model_comparison.csv`, `data/svm_tuning_results.csv` — recorded evaluation results

## App walkthrough

The screenshots below are intended to document the main user journey once the image files are committed under `docs/images/`:

1. **Landing page** — introduce MoodTunes and the mood-entry form.
2. **Mood input** — enter a short description of how you are feeling.
3. **Emotion and listening mood** — see the predicted emotion alongside the mapped listening mood (for example, Sadness → Calm).
4. **Recommendations** — browse the tracks selected for that listening mood.

The screenshots should be kept in sync with the current Streamlit UI and should show genuine application output, not mockups.

## Limitations and next steps

- Emotion classification is an estimate from text, not a clinical or definitive assessment of a person's mood.
- Song recommendations are sampled from the local catalog by mapped mood; they are not personalized from listening history.
- Record the dataset's original citation and license.
- Add automated tests for preprocessing, prediction, and recommendation mapping.
- Capture real application screenshots and document the dataset split sizes and evaluation setup.

## License

See [LICENSE](LICENSE) for the repository license.

---
Built by [Vishakha Khetan](https://github.com/kVishakhha).
