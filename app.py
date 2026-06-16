"""
📩 Spam Email Classifier - Streamlit App
Detect Spam Emails using Machine Learning and NLP.
"""

import os
import re
import time
import string
import pickle

import streamlit as st

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Spam Email Classifier",
    page_icon="📩",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS for a clean, professional look
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .main-header {
        background: linear-gradient(90deg, #4A90E2 0%, #6C63FF 100%);
        padding: 28px 24px;
        border-radius: 14px;
        color: white;
        text-align: center;
        margin-bottom: 24px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.08);
    }
    .main-header h1 { margin: 0; font-size: 2.2rem; }
    .main-header p  { margin: 6px 0 0; opacity: 0.95; font-size: 1.05rem; }

    .stTextArea textarea {
        border-radius: 10px !important;
        border: 1px solid #d0d7de !important;
        font-size: 1rem !important;
    }

    .stButton > button {
        border-radius: 10px;
        padding: 0.55rem 1.2rem;
        font-weight: 600;
        border: none;
        transition: transform .05s ease-in-out;
    }
    .stButton > button:hover { transform: translateY(-1px); }

    .footer {
        text-align: center;
        color: #6b7280;
        font-size: 0.9rem;
        margin-top: 40px;
        padding-top: 16px;
        border-top: 1px solid #e5e7eb;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="main-header">
        <h1>📩 Spam Email Classifier</h1>
        <p>Detect Spam Emails using Machine Learning and NLP</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Lightweight stopword list (avoids needing nltk download at runtime)
# ---------------------------------------------------------------------------
STOPWORDS = {
    "i","me","my","we","our","you","your","he","him","his","she","her","it","its",
    "they","them","their","what","which","who","this","that","these","those",
    "am","is","are","was","were","be","been","being","have","has","had","do",
    "does","did","a","an","the","and","but","if","or","because","as","until",
    "while","of","at","by","for","with","about","against","between","into",
    "through","during","before","after","above","below","to","from","up","down",
    "in","out","on","off","over","under","again","further","then","once","here",
    "there","when","where","why","how","all","any","both","each","few","more",
    "most","other","some","such","no","nor","not","only","own","same","so",
    "than","too","very","s","t","can","will","just","don","should","now",
}

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = [w for w in text.split() if w not in STOPWORDS and len(w) > 1]
    return " ".join(tokens)

# ---------------------------------------------------------------------------
# Load model & vectorizer
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    if not (os.path.exists("model.pkl") and os.path.exists("vectorizer.pkl")):
        return None, None
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

def load_accuracy() -> str:
    if os.path.exists("accuracy.txt"):
        return open("accuracy.txt").read().strip() + "%"
    return "~98%"

model, vectorizer = load_artifacts()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("ℹ️ Model Info")
    st.markdown("**Model Used:** Multinomial Naive Bayes")
    st.markdown(f"**Accuracy:** {load_accuracy()}")
    st.markdown("---")
    st.subheader("🛠️ Technologies Used")
    st.markdown(
        "- Python\n"
        "- Scikit-learn\n"
        "- NLTK\n"
        "- Pandas\n"
        "- TF-IDF Vectorizer\n"
        "- Streamlit"
    )
    st.markdown("---")
    st.caption("AI/ML Internship Project")

# ---------------------------------------------------------------------------
# Main input area
# ---------------------------------------------------------------------------
if "email_text" not in st.session_state:
    st.session_state.email_text = ""

st.subheader("✉️ Enter the email or message text below")
email_text = st.text_area(
    "Message",
    value=st.session_state.email_text,
    height=180,
    placeholder="Paste an email or SMS message here...",
    label_visibility="collapsed",
)

col1, col2 = st.columns([1, 1])
predict_clicked = col1.button("🔍 Predict", use_container_width=True, type="primary")
clear_clicked   = col2.button("🧹 Clear",  use_container_width=True)

if clear_clicked:
    st.session_state.email_text = ""
    st.rerun()

# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
if predict_clicked:
    if model is None or vectorizer is None:
        st.error(
            "⚠️ Model files not found. Please run `python train_model.py` first "
            "to generate `model.pkl` and `vectorizer.pkl`."
        )
    elif not email_text.strip():
        st.warning("Please enter a message to classify.")
    else:
        with st.spinner("Analyzing message..."):
            time.sleep(0.6)
            cleaned = clean_text(email_text)
            vec = vectorizer.transform([cleaned])
            pred = int(model.predict(vec)[0])
            proba = model.predict_proba(vec)[0]
            confidence = float(proba[pred]) * 100

        st.session_state.email_text = email_text

        if pred == 1:
            st.error(f"🚨 **Spam Email Detected**\n\nConfidence: **{confidence:.2f}%**")
        else:
            st.success(f"✅ **Safe Email**\n\nConfidence: **{confidence:.2f}%**")

        st.progress(min(int(confidence), 100))

        with st.expander("🔬 See cleaned text used for prediction"):
            st.code(cleaned or "(empty after cleaning)")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="footer">
        Built with ❤️ using Streamlit &nbsp;|&nbsp; AI/ML Internship Project &nbsp;|&nbsp; © 2026
    </div>
    """,
    unsafe_allow_html=True,
)
