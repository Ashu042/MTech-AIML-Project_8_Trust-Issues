# 🕵️ Fake Product Review Detector

A machine learning system that flags likely-fake product reviews, built
entirely with free/open-source tools and designed to deploy for free in
minutes.

## How it works

**Two feature streams, combined:**

1. **TF-IDF text features** (word 1–2 grams) — captures word/phrase patterns.
2. **Hand-crafted linguistic + behavioral features** (`features.py`), based
   on patterns documented in deceptive-text research (Ott et al. 2011;
   Mukherjee et al. 2013):
   - Exclamation-mark density, ALL-CAPS ratio
   - Superlative density ("best ever", "perfect", "life-changing")
   - Sentiment score & subjectivity (lightweight lexicon, no downloads)
   - Concrete-detail ratio (mentions of sizes, hours, materials, etc. —
     genuine reviews tend to have more of these; fake ones are vaguer)
   - First-person pronoun ratio, unique-word ratio (repetition)
   - **Rating/sentiment mismatch** — e.g. a 5-star review written with
     negative language, a classic red flag
   - Verified-purchase flag, reviewer review-count ("thin history")

**Model:** a soft-voting ensemble of Logistic Regression + Random Forest
(`train_model.py`), which tends to be both accurate and reasonably
well-calibrated for probability outputs.

**Output:** for any review you get a fake-probability score (0–100%) *and*
a plain-English list of which signals triggered — not just a black-box
label — so you can sanity-check the verdict yourself.

## ⚠️ Important note on the training data

`generate_dataset.py` builds a large **synthetic** dataset from templated
patterns grounded in the research above. This lets the whole pipeline work
out of the box with zero setup and no API keys/logins — but templated data
is easier to separate than the real world, so treat the current model as a
**working demo/scaffold**, not a production-accuracy classifier.

To get real-world accuracy:
1. Get a real labeled dataset — e.g. the "Fake Reviews Dataset" on Kaggle,
   or the Ott et al. Deceptive Opinion Spam Corpus — and export it as a
   CSV with columns: `text, rating, verified_purchase, reviewer_review_count, label`
   (label: 1 = fake, 0 = genuine).
2. Replace `reviews_dataset.csv` with that file.
3. Re-run `python3 train_model.py`. No other code changes needed.

## Project structure

```
fake_review_detector/
├── generate_dataset.py   # builds reviews_dataset.csv (swap for real data)
├── features.py           # linguistic/behavioral feature extraction
├── train_model.py        # trains the ensemble, saves model.joblib
├── predict.py            # inference + human-readable explanations
├── app.py                # Streamlit web UI
├── requirements.txt
└── model.joblib           # trained pipeline (generated)
```

## Run it locally

```bash
pip install -r requirements.txt
python3 generate_dataset.py   # only needed once, or to regenerate data
python3 train_model.py        # trains model.joblib (~10 sec)
streamlit run app.py          # opens the web app at localhost:8501
```

## Deploy for free (pick one)

### Option A — Streamlit Community Cloud (easiest)
1. Push this folder to a public (or private) GitHub repo, **including**
   `model.joblib` (or re-run `train_model.py` as a build step).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with
   GitHub, click "New app", pick the repo and `app.py` as the entry point.
3. Deploy — you get a free public URL in ~2 minutes.

### Option B — Hugging Face Spaces (also free)
1. Create a new Space at [huggingface.co/spaces](https://huggingface.co/spaces),
   choose the **Streamlit** SDK.
2. Upload/push the same files (`app.py`, `predict.py`, `features.py`,
   `requirements.txt`, `model.joblib`).
3. The Space builds and hosts it automatically at a public URL.

### Option C — Render / Railway free tier
Add a `Procfile` with:
```
web: streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```
and connect the repo — both platforms have free tiers suitable for a
lightweight app like this.

## Using it programmatically

```python
from predict import predict_review

result = predict_review(
    "AMAZING!!! Best product EVER, buy it now, changed my life!!!",
    rating=5,
    verified_purchase=0,
    reviewer_review_count=1,
)
print(result)
# {'label': 'fake', 'fake_probability': 0.98, 'confidence': 0.96,
#  'reasons': ['Unusually high use of exclamation marks', ...]}
```

## Extending it

- **More accuracy:** swap TF-IDF for free sentence embeddings
  (`sentence-transformers`, e.g. `all-MiniLM-L6-v2`) — still free, just a
  slightly heavier install.
- **Reviewer-network signals:** if you have access to a reviewer's full
  history, add features like posting-burst detection (many reviews in a
  short window) and cross-review similarity (near-duplicate text across
  different products), both strong fraud signals in the Mukherjee et al.
  research.
- **Active learning:** log low-confidence predictions (0.4–0.6 probability)
  from real usage and periodically relabel/retrain on them.
