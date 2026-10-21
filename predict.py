"""
predict.py
----------
Loads the trained pipeline and exposes predict_review() for single or
batch inference. Used by app.py (Streamlit) and can be imported directly.
"""

import joblib
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from features import extract_features_df, extract_text_features

_ARTIFACT = None


def _load():
    global _ARTIFACT
    if _ARTIFACT is None:
        _ARTIFACT = joblib.load("model.joblib")
    return _ARTIFACT


def predict_review(text: str, rating: int = 5, verified_purchase: int = 1,
                    reviewer_review_count: int = 10) -> dict:
    """
    Returns a dict with:
      - label: 'fake' or 'genuine'
      - fake_probability: float 0-1
      - confidence: float 0-1 (distance from 0.5, rescaled)
      - reasons: list of human-readable signal explanations
    """
    art = _load()
    df = pd.DataFrame([{
        "text": text,
        "rating": rating,
        "verified_purchase": verified_purchase,
        "reviewer_review_count": reviewer_review_count,
    }])

    X_tfidf = art["tfidf"].transform(df["text"])
    feats = extract_features_df(df)[art["numeric_feature_columns"]]
    feats_scaled = art["scaler"].transform(feats)
    X = hstack([X_tfidf, csr_matrix(feats_scaled)]).tocsr()

    proba_fake = float(art["model"].predict_proba(X)[0, 1])
    label = "fake" if proba_fake >= 0.5 else "genuine"
    confidence = abs(proba_fake - 0.5) * 2

    reasons = _explain(feats.iloc[0].to_dict(), proba_fake)

    return {
        "label": label,
        "fake_probability": round(proba_fake, 4),
        "confidence": round(confidence, 4),
        "reasons": reasons,
    }


def _explain(f: dict, proba_fake: float) -> list:
    reasons = []
    if f["exclamation_density"] > 0.03:
        reasons.append("Unusually high use of exclamation marks")
    if f["superlative_density"] > 0.02:
        reasons.append("Heavy use of superlatives (e.g. 'best ever', 'perfect')")
    if f["caps_word_ratio"] > 0.03:
        reasons.append("Frequent ALL-CAPS words")
    if f["concrete_detail_ratio"] < 0.01 and f["word_count"] > 8:
        reasons.append("Lacks specific, concrete product details")
    if f["rating_sentiment_mismatch"] > 0.6:
        reasons.append("Star rating doesn't match the sentiment of the text")
    if f["verified_purchase"] == 0:
        reasons.append("Not a verified purchase")
    if f["thin_history"] == 1:
        reasons.append("Reviewer has very little review history")
    if f["extreme_rating"] == 1:
        reasons.append("Extreme rating (1 or 5 stars)")
    if f["unique_word_ratio"] < 0.55 and f["word_count"] > 15:
        reasons.append("Repetitive word choice")
    if f["word_count"] < 8:
        reasons.append("Very short review with little substantive content")

    if not reasons:
        if proba_fake < 0.5:
            reasons.append("Balanced sentiment with specific, concrete details")
        else:
            reasons.append("Overall language pattern statistically resembles fake reviews")
    return reasons


if __name__ == "__main__":
    examples = [
        "AMAZING!!! Best product EVER, exceeded all my expectations, perfect quality, buy it now you won't regret it!!!",
        "I've had these earbuds for about 3 weeks. Battery lasts around 6 hours which is fine for my commute. The case scratches easily but the sound quality is solid for the price.",
    ]
    for ex in examples:
        print(ex[:60], "...")
        print(predict_review(ex))
        print()
