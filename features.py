"""
features.py
------------
Hand-crafted linguistic / stylometric / behavioral features for fake
review detection, based on patterns documented in deceptive-text and
opinion-spam research (Ott et al. 2011; Mukherjee et al. 2013; Jindal &
Liu 2008). No external corpus downloads required -> deploys anywhere.

Feature groups:
  1. Surface stats       -> length, word count, avg word length
  2. Punctuation/caps     -> exclamation density, ALL-CAPS ratio, ellipses
  3. Sentiment/subjectivity (lexicon-based, self-contained)
  4. Specificity          -> numbers, concrete nouns proxy, pronoun ratio
  5. Repetition            -> unique-word ratio, superlative density
  6. Rating/text mismatch -> sentiment vs star rating disagreement
  7. Behavioral (metadata)-> verified purchase, reviewer history depth
"""

import re
import numpy as np
import pandas as pd

# ---- small self-contained lexicons (no downloads) ----
POSITIVE_WORDS = set("""
good great excellent amazing perfect love loved loving best awesome fantastic
wonderful nice happy satisfied recommend recommended quality worth impressed
solid comfortable easy smooth reliable durable pleased fast quick great
""".split())

NEGATIVE_WORDS = set("""
bad terrible awful worst hate hated broke broken disappointed disappointing
cheap flimsy slow waste poor uncomfortable defective useless problem issue
returned refund regret junk garbage
""".split())

SUPERLATIVES = set("""
best worst amazing incredible perfect never everyone always life-changing
unbelievable phenomenal outstanding flawless
""".split())

FIRST_PERSON = set(["i", "my", "me", "mine", "myself"])
CONCRETE_HINTS = set("""
inch inches cm mm oz lb kg hour hours minute minutes day days week weeks
month months year years size battery charge charger button screen material
plastic metal fabric handle cord cable warranty manual box packaging weight
""".split())


def _tokenize(text: str):
    return re.findall(r"[a-zA-Z']+", text.lower())


def extract_text_features(text: str) -> dict:
    text = text if isinstance(text, str) else ""
    words = _tokenize(text)
    n_words = max(len(words), 1)
    n_chars = max(len(text), 1)
    sentences = max(len(re.findall(r"[.!?]+", text)), 1)

    exclam = text.count("!")
    caps_words = sum(1 for w in re.findall(r"[A-Za-z']+", text) if len(w) > 2 and w.isupper())
    digits = sum(1 for c in text if c.isdigit())

    pos_hits = sum(1 for w in words if w in POSITIVE_WORDS)
    neg_hits = sum(1 for w in words if w in NEGATIVE_WORDS)
    superlative_hits = sum(1 for w in words if w in SUPERLATIVES)
    first_person_hits = sum(1 for w in words if w in FIRST_PERSON)
    concrete_hits = sum(1 for w in words if w in CONCRETE_HINTS)

    unique_ratio = len(set(words)) / n_words
    avg_word_len = sum(len(w) for w in words) / n_words
    sentiment_score = (pos_hits - neg_hits) / n_words
    subjectivity = (pos_hits + neg_hits) / n_words

    return {
        "char_count": n_chars,
        "word_count": n_words,
        "avg_word_len": avg_word_len,
        "sentence_count": sentences,
        "avg_words_per_sentence": n_words / sentences,
        "exclamation_density": exclam / n_words,
        "caps_word_ratio": caps_words / n_words,
        "digit_ratio": digits / n_chars,
        "unique_word_ratio": unique_ratio,
        "sentiment_score": sentiment_score,
        "subjectivity": subjectivity,
        "superlative_density": superlative_hits / n_words,
        "first_person_ratio": first_person_hits / n_words,
        "concrete_detail_ratio": concrete_hits / n_words,
        "pos_word_count": pos_hits,
        "neg_word_count": neg_hits,
    }


def extract_features_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    df must have columns: text, rating, verified_purchase, reviewer_review_count
    Returns a DataFrame of numeric features (text feats + metadata + rating/sentiment mismatch).
    """
    feat_rows = [extract_text_features(t) for t in df["text"]]
    feats = pd.DataFrame(feat_rows)

    feats["verified_purchase"] = df["verified_purchase"].values
    feats["reviewer_review_count"] = df["reviewer_review_count"].values
    feats["rating"] = df["rating"].values

    # rating/sentiment mismatch: a 5-star review with negative wording, or
    # a 1-star review with positive wording, is a red flag
    normalized_rating_sentiment = (feats["rating"] - 3) / 2.0  # maps 1..5 -> -1..1
    feats["rating_sentiment_mismatch"] = (
        normalized_rating_sentiment - feats["sentiment_score"].clip(-1, 1)
    ).abs()

    # extreme rating flag (1 or 5 stars are more common in fake reviews)
    feats["extreme_rating"] = feats["rating"].isin([1, 5]).astype(int)

    # thin reviewer history flag
    feats["thin_history"] = (feats["reviewer_review_count"] <= 2).astype(int)

    return feats


NUMERIC_FEATURE_COLUMNS = [
    "char_count", "word_count", "avg_word_len", "sentence_count",
    "avg_words_per_sentence", "exclamation_density", "caps_word_ratio",
    "digit_ratio", "unique_word_ratio", "sentiment_score", "subjectivity",
    "superlative_density", "first_person_ratio", "concrete_detail_ratio",
    "pos_word_count", "neg_word_count", "verified_purchase",
    "reviewer_review_count", "rating", "rating_sentiment_mismatch",
    "extreme_rating", "thin_history",
]
