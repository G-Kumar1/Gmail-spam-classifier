import streamlit as st
import joblib
import json
from pathlib import Path
import numpy as np

from imap_gmail import fetch_recent_emails

MODEL_DIR = Path("models")


# ========== LOAD MODEL & VECTORIZER ==========
@st.cache_resource
def load_model_and_vectorizer():
    model = joblib.load(MODEL_DIR / "spam_classifier.joblib")
    vectorizer = joblib.load(MODEL_DIR / "tfidf_vectorizer.joblib")
    return model, vectorizer


# ========== LOAD METRICS ==========
@st.cache_data
def load_metrics():
    metrics_path = MODEL_DIR / "metrics.json"
    if not metrics_path.exists():
        return None
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    return metrics


# ========== THRESHOLD-AWARE CLASSIFICATION ==========
def classify_emails(emails, model, vectorizer, threshold):
    texts = [e["text"] for e in emails]
    X_vec = vectorizer.transform(texts)

    # Get spam probabilities
    proba = model.predict_proba(X_vec)[:, 1]

    # Apply user-defined threshold
    preds = (proba >= threshold).astype(int)

    for i, email in enumerate(emails):
        email["pred_label"] = int(preds[i])
        email["pred_label_text"] = "SPAM" if preds[i] == 1 else "NOT SPAM"
        email["spam_score"] = float(proba[i])

    return emails


# ===================== STREAMLIT UI =====================
def main():
    st.set_page_config(page_title="Gmail Spam Classifier", layout="wide")
    st.title("📧 Gmail Spam Classifier")

    # ---------- SIDEBAR ----------
    st.sidebar.header("📊 Model Performance")
    metrics = load_metrics()

    if metrics:
        st.sidebar.metric("Accuracy", f"{metrics['accuracy']:.3f}")
        st.sidebar.metric("Precision", f"{metrics['precision']:.3f}")
        st.sidebar.metric("Recall", f"{metrics['recall']:.3f}")
        st.sidebar.metric("F1 Score", f"{metrics['f1']:.3f}")
    else:
        st.sidebar.warning("No metrics.json found. Train the model first.")

    # ✅ USER-DEFINED THRESHOLD
    st.sidebar.header("⚙️ Prediction Control")
    threshold = st.sidebar.slider(
        "Spam Decision Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.7,
        step=0.05
    )

    num_emails = st.sidebar.slider(
        "Number of recent emails to fetch",
        min_value=5,
        max_value=50,
        value=15,
        step=5
    )

    model, vectorizer = load_model_and_vectorizer()

    st.write(
        "Click the button below to classify your recent Gmail emails as SPAM or NOT SPAM."
    )

    # ---------- FETCH & CLASSIFY ----------
    if st.button("📥 Fetch & Classify Emails"):
        with st.spinner("Connecting to Gmail and classifying emails..."):
            emails = fetch_recent_emails(limit=num_emails)

            if not emails:
                st.warning("No emails fetched from Gmail.")
                return

            emails = classify_emails(emails, model, vectorizer, threshold)

        spam_count = sum(1 for e in emails if e["pred_label"] == 1)

        st.success(f"Fetched {len(emails)} emails | {spam_count} marked as SPAM")

        # ---------- SUMMARY ----------
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Emails", len(emails))
        with col2:
            st.metric("Spam", spam_count)
        with col3:
            st.metric("Not Spam", len(emails) - spam_count)

        # ---------- EMAIL DISPLAY ----------
        for email in emails:
            label = "🔴 SPAM" if email["pred_label"] == 1 else "🟢 NOT SPAM"

            with st.expander(f"{label} | {email['subject']}"):
                st.write(f"**From:** {email['from']}")
                st.write(f"**Spam Probability:** {email['spam_score']:.3f}")
                st.write("---")
                st.write(email["body"][:5000])


if __name__ == "__main__":
    main()
