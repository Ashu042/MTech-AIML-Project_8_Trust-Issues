"""
app.py — Streamlit UI for the Fake Product Review Detector.

Run locally:   streamlit run app.py
Deploy free:   push this folder to GitHub, then deploy on
               Streamlit Community Cloud (share.streamlit.io) or
               Hugging Face Spaces (Streamlit SDK) — both free.
"""

import streamlit as st
import pandas as pd
from predict import predict_review

st.set_page_config(page_title="Fake Review Detector", page_icon="🕵️", layout="centered")

st.title("🕵️ Fake Product Review Detector")
st.caption(
    "Paste a review and get a fake/genuine probability, based on linguistic "
    "patterns, sentiment/rating consistency, and reviewer behavior signals."
)

tab1, tab2 = st.tabs(["🔍 Check a single review", "📂 Batch check (CSV)"])

with tab1:
    text = st.text_area(
        "Review text",
        height=150,
        placeholder="e.g. 'AMAZING!!! Best product ever, changed my life, buy it now!!!'",
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        rating = st.slider("Star rating given", 1, 5, 5)
    with col2:
        verified = st.selectbox("Verified purchase?", ["Yes", "No"])
    with col3:
        history = st.number_input("Reviewer's past review count", min_value=0, value=10)

    if st.button("Analyze review", type="primary", use_container_width=True):
        if not text.strip():
            st.warning("Please paste a review first.")
        else:
            result = predict_review(
                text,
                rating=rating,
                verified_purchase=1 if verified == "Yes" else 0,
                reviewer_review_count=history,
            )
            proba = result["fake_probability"]

            if result["label"] == "fake":
                st.error(f"⚠️ Likely FAKE — {proba*100:.1f}% probability")
            else:
                st.success(f"✅ Likely GENUINE — {(1-proba)*100:.1f}% probability of being genuine")

            st.progress(proba, text=f"Fake probability: {proba*100:.1f}%")

            st.markdown("**Signals detected:**")
            for r in result["reasons"]:
                st.markdown(f"- {r}")

            with st.expander("Model confidence detail"):
                st.write(f"Confidence: {result['confidence']*100:.1f}%")
                st.caption(
                    "Confidence reflects how far the prediction is from the "
                    "50/50 decision boundary, not overall model accuracy."
                )

with tab2:
    st.write("Upload a CSV with a `text` column (optional: `rating`, `verified_purchase`, `reviewer_review_count`).")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        if "text" not in df.columns:
            st.error("CSV must have a 'text' column.")
        else:
            df["rating"] = df.get("rating", pd.Series([5] * len(df)))
            df["verified_purchase"] = df.get("verified_purchase", pd.Series([1] * len(df)))
            df["reviewer_review_count"] = df.get("reviewer_review_count", pd.Series([10] * len(df)))

            with st.spinner("Analyzing reviews..."):
                results = [
                    predict_review(
                        row["text"], row["rating"], row["verified_purchase"], row["reviewer_review_count"]
                    )
                    for _, row in df.iterrows()
                ]
            df["prediction"] = [r["label"] for r in results]
            df["fake_probability"] = [r["fake_probability"] for r in results]

            st.dataframe(df[["text", "prediction", "fake_probability"]], use_container_width=True)
            st.download_button(
                "Download results as CSV",
                df.to_csv(index=False).encode("utf-8"),
                "review_predictions.csv",
                "text/csv",
            )

st.divider()
st.caption(
    "⚠️ This is a decision-support tool trained on synthetic patterns from published "
    "fake-review research — treat outputs as a probability signal, not ground truth. "
    "For production accuracy, retrain on real labeled review data (see README)."
)
