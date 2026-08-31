import os
import re
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Lowercase
    text = text.lower()
    # Remove user mentions @username
    text = re.sub(r"@[A-Za-z0-9_]+", "", text)
    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    # Keep letters, numbers, basic punctuation like ! and ?
    text = re.sub(r"[^\w\s!?']", " ", text)
    # Collapse multiple whitespaces
    text = re.sub(r"\s+", " ", text).strip()
    return text

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "data", "emotion_dataset_raw.csv")
    model_dir = os.path.join(base_dir, "model")
    model_path = os.path.join(model_dir, "text_emotion.pkl")
    cm_output_path = os.path.join(base_dir, "confusion_matrix.png")

    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Dataset shape: {df.shape}")
    print("\nClass distribution:\n", df['Emotion'].value_counts())

    # Preprocessing
    print("\nPreprocessing text data...")
    df['Clean_Text'] = df['Text'].apply(clean_text)

    # Filter out any empty rows after cleaning
    df = df[df['Clean_Text'].str.strip().str.len() > 0].copy()

    X = df['Clean_Text']
    y = df['Emotion']

    # Stratified Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain samples: {len(X_train)}, Test samples: {len(X_test)}")

    # Pipeline with TF-IDF (unigrams + bigrams) and Balanced Logistic Regression
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_features=30000,
            sublinear_tf=True
        )),
        ('lr', LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            C=1.5,
            solver='lbfgs'
        ))
    ])

    print("\nTraining pipeline...")
    pipeline.fit(X_train, y_train)

    # Evaluation
    print("\nEvaluating model on test split...")
    y_pred = pipeline.predict(X_test)
    labels = sorted(list(y.unique()))

    report = classification_report(y_test, y_pred, target_names=labels, digits=4)
    print("\n" + "=" * 65)
    print("                CLASSIFICATION REPORT")
    print("=" * 65)
    print(report)
    print("=" * 65)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels
    )
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Emotion Classification Confusion Matrix (Balanced TF-IDF + Logistic Regression)', fontsize=13)
    plt.tight_layout()
    plt.savefig(cm_output_path, dpi=300)
    plt.close()
    print(f"\nConfusion matrix saved to: {cm_output_path}")

    # Save model
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(pipeline, model_path)
    print(f"Trained model saved to: {model_path}")

    # Test sample phrases
    print("\n" + "=" * 65)
    print("                  INFERENCE DEMO")
    print("=" * 65)
    test_samples = [
        "i am hungry",
        "I just won the lottery! Best day ever!",
        "I am so angry and furious with this terrible service",
        "I am not happy about what happened",
        "Tomorrow is Tuesday.",
        "I am terrified of heights"
    ]
    for sample in test_samples:
        cleaned = clean_text(sample)
        proba = pipeline.predict_proba([cleaned])[0]
        max_idx = np.argmax(proba)
        pred = pipeline.classes_[max_idx]
        conf = proba[max_idx]
        print(f"Text: '{sample}'")
        print(f" -> Predicted: {pred} (Confidence: {conf:.4f})")
        top_3 = sorted(zip(pipeline.classes_, proba), key=lambda x: x[1], reverse=True)[:3]
        print(f" -> Top 3: {[(c, round(p, 3)) for c, p in top_3]}")
        print("-" * 50)

if __name__ == "__main__":
    main()
