"""Tune LinearSVC on validation data, then evaluate the winner once on test."""
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.svm import LinearSVC

from model_pipeline import make_vectorizer, preprocess_text


ROOT = Path(__file__).resolve().parents[1]
LABEL_MAP = {0: "sadness", 1: "joy", 2: "love", 3: "anger", 4: "fear", 5: "surprise"}
C_VALUES = (0.1, 0.5, 1, 2, 5)


def load_data(name):
    frame = pd.read_csv(ROOT / "data" / f"{name}.csv")
    frame["emotion"] = frame["label"].map(LABEL_MAP)
    return frame


def main():
    train_df = load_data("training")
    validation_df = load_data("validation")
    train_text = train_df["text"].map(preprocess_text)
    validation_text = validation_df["text"].map(preprocess_text)

    vectorizer = make_vectorizer()
    train_x = vectorizer.fit_transform(train_text)
    validation_x = vectorizer.transform(validation_text)
    y_train = train_df["emotion"]
    y_validation = validation_df["emotion"]

    results = []
    for c_value in C_VALUES:
        model = LinearSVC(C=c_value, class_weight="balanced")
        model.fit(train_x, y_train)
        predictions = model.predict(validation_x)
        results.append({
            "C": c_value,
            "Accuracy": accuracy_score(y_validation, predictions),
            "Macro F1": f1_score(y_validation, predictions, average="macro"),
            "Weighted F1": f1_score(y_validation, predictions, average="weighted"),
        })

    results_df = pd.DataFrame(results).sort_values(
        ["Macro F1", "C"], ascending=[False, True], kind="stable"
    )
    results_df.to_csv(ROOT / "data" / "svm_tuning_results.csv", index=False)
    best = results_df.iloc[0]
    best_c = float(best["C"])
    print(f"Best C (selected by validation Macro F1): {best_c:g}")
    print(
        "Validation metrics: "
        f"Accuracy={best['Accuracy']:.4f}, "
        f"Macro F1={best['Macro F1']:.4f}, "
        f"Weighted F1={best['Weighted F1']:.4f}"
    )
    print(f"Tuning results saved to {ROOT / 'data' / 'svm_tuning_results.csv'}")

    # The test split is loaded and used only after validation has selected C.
    test_df = load_data("test")
    test_text = test_df["text"].map(preprocess_text)
    test_x = vectorizer.transform(test_text)
    final_model = LinearSVC(C=best_c, class_weight="balanced")
    final_model.fit(train_x, y_train)
    y_test = test_df["emotion"]
    test_predictions = final_model.predict(test_x)

    print("\nFinal test evaluation (weighted Precision/Recall/F1):")
    print(f"Accuracy:  {accuracy_score(y_test, test_predictions):.4f}")
    print(f"Precision: {precision_score(y_test, test_predictions, average='weighted', zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_test, test_predictions, average='weighted', zero_division=0):.4f}")
    print(f"F1:        {f1_score(y_test, test_predictions, average='weighted', zero_division=0):.4f}")
    labels = [LABEL_MAP[index] for index in sorted(LABEL_MAP)]
    print("\nClassification report:")
    print(classification_report(y_test, test_predictions, labels=labels, zero_division=0))
    print("Confusion matrix (rows=true, columns=predicted; label order shown above):")
    print(confusion_matrix(y_test, test_predictions, labels=labels))
    print("Label order:", labels)


if __name__ == "__main__":
    main()
