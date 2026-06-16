"""
Spam Email Classifier - Model Training Script
Trains a Multinomial Naive Bayes model on the SMS Spam Collection Dataset.
Saves model.pkl and vectorizer.pkl for use by the Streamlit app.
"""

import os
import re
import string
import pickle
import urllib.request
import zipfile
import io

import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# ---------------------------------------------------------------------------
# 1. Download NLTK stopwords (only first time)
# ---------------------------------------------------------------------------
try:
    stopwords.words("english")
except LookupError:
    nltk.download("stopwords")

STOPWORDS = set(stopwords.words("english"))


# ---------------------------------------------------------------------------
# 2. Load dataset (auto-download if missing)
# ---------------------------------------------------------------------------
DATA_FILE = "spam.csv"


def load_dataset() -> pd.DataFrame:
    if not os.path.exists(DATA_FILE):
        print("📥 Downloading SMS Spam Collection Dataset...")
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
        with urllib.request.urlopen(url) as resp:
            data = resp.read()
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            with zf.open("SMSSpamCollection") as f:
                df = pd.read_csv(f, sep="\t", header=None, names=["label", "message"])
        df.to_csv(DATA_FILE, index=False)
    else:
        df = pd.read_csv(DATA_FILE)
    return df


# ---------------------------------------------------------------------------
# 3. Text cleaning
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)        # remove urls
    text = re.sub(r"\d+", " ", text)                      # remove digits
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = [w for w in text.split() if w not in STOPWORDS and len(w) > 1]
    return " ".join(tokens)


# ---------------------------------------------------------------------------
# 4. Main training pipeline
# ---------------------------------------------------------------------------
def main() -> None:
    print("📊 Loading dataset...")
    df = load_dataset()
    print(f"   Total samples: {len(df)}")

    print("🧹 Cleaning text...")
    df["clean"] = df["message"].apply(clean_text)
    df["target"] = df["label"].map({"ham": 0, "spam": 1})

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean"], df["target"], test_size=0.2, random_state=42, stratify=df["target"]
    )

    print("🔠 Applying TF-IDF Vectorization...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print("🤖 Training Multinomial Naive Bayes...")
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    # -------- Evaluation --------
    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)

    print("\n========== EVALUATION ==========")
    print(f"Accuracy: {acc * 100:.2f}%")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Ham", "Spam"]))

    # -------- Save artifacts --------
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    # Save accuracy for the Streamlit sidebar
    with open("accuracy.txt", "w") as f:
        f.write(f"{acc * 100:.2f}")

    print("\n✅ Saved: model.pkl, vectorizer.pkl, accuracy.txt")


if __name__ == "__main__":
    main()
