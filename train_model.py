"""
train_model.py
---------------
Trains the fake-review detection model:
  - TF-IDF (word 1-2 grams) on review text
  - Hand-crafted linguistic/behavioral features (features.py)
  - Combined via a VotingClassifier ensemble (Logistic Regression +
    Random Forest) for both accuracy and reasonable calibration.

Run: python3 train_model.py
Outputs: model.joblib (contains the full pipeline, ready for inference)
"""

import pandas as pd
import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import joblib

from features import extract_features_df, NUMERIC_FEATURE_COLUMNS


def load_data(path="reviews_dataset.csv"):
    return pd.read_csv(path)


def build_and_train(df: pd.DataFrame):
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        df, df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    # ---- text features (TF-IDF) ----
    tfidf = TfidfVectorizer(
        max_features=3000, ngram_range=(1, 2), min_df=2, stop_words="english"
    )
    X_train_tfidf = tfidf.fit_transform(X_train_df["text"])
    X_test_tfidf = tfidf.transform(X_test_df["text"])

    # ---- engineered numeric features ----
    train_feats = extract_features_df(X_train_df)[NUMERIC_FEATURE_COLUMNS]
    test_feats = extract_features_df(X_test_df)[NUMERIC_FEATURE_COLUMNS]

    scaler = StandardScaler()
    train_feats_scaled = scaler.fit_transform(train_feats)
    test_feats_scaled = scaler.transform(test_feats)

    # ---- combine sparse TF-IDF + dense engineered features ----
    X_train = hstack([X_train_tfidf, csr_matrix(train_feats_scaled)]).tocsr()
    X_test = hstack([X_test_tfidf, csr_matrix(test_feats_scaled)]).tocsr()

    # ---- ensemble model ----
    log_reg = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=None, min_samples_leaf=2,
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    ensemble = VotingClassifier(
        estimators=[("logreg", log_reg), ("rf", rf)], voting="soft", weights=[1, 1.2]
    )
    ensemble.fit(X_train, y_train)

    # ---- evaluation ----
    y_pred = ensemble.predict(X_test)
    y_proba = ensemble.predict_proba(X_test)[:, 1]

    print("=== Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=["genuine", "fake"]))
    print("ROC-AUC:", round(roc_auc_score(y_test, y_proba), 4))
    print("Confusion matrix [[TN FP][FN TP]]:")
    print(confusion_matrix(y_test, y_pred))

    artifact = {
        "tfidf": tfidf,
        "scaler": scaler,
        "model": ensemble,
        "numeric_feature_columns": NUMERIC_FEATURE_COLUMNS,
    }
    joblib.dump(artifact, "model.joblib")
    print("\nSaved trained pipeline to model.joblib")
    return artifact


if __name__ == "__main__":
    df = load_data()
    build_and_train(df)
