# ==========================================================
# Spam Email Detection - Streamlit Application
# ==========================================================

import streamlit as st
import joblib
import sys
import time

sys.path.append("src")
from preprocessing import transform_text


# ----------------------------------------------------------
# Page Configuration (must be first Streamlit command)
# ----------------------------------------------------------

st.set_page_config(
    page_title="Spam Email Detector",
    page_icon="📧",
    layout="centered",
    initial_sidebar_state="expanded"
)


# ----------------------------------------------------------
# Custom CSS
# ----------------------------------------------------------

st.markdown(
    """
    <style>
        /* Overall app background */
        .stApp {
            background: linear-gradient(180deg, #f7f9fc 0%, #eef1f8 100%);
        }

        /* Main title */
        .main-title {
            font-size: 2.4rem;
            font-weight: 800;
            text-align: center;
            margin-bottom: 0.2rem;
            background: linear-gradient(90deg, #4f46e5, #7c3aed);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .subtitle {
            text-align: center;
            color: #6b7280;
            font-size: 1.05rem;
            margin-bottom: 1.8rem;
        }

        /* Card container */
        .card {
            background: white;
            border-radius: 16px;
            padding: 1.4rem 1.6rem;
            box-shadow: 0 4px 18px rgba(0,0,0,0.06);
            margin-bottom: 1.2rem;
            border: 1px solid #eef0f4;
        }

        /* Result banners */
        .result-spam {
            background: linear-gradient(90deg, #fee2e2, #fecaca);
            border-left: 6px solid #dc2626;
            padding: 1rem 1.2rem;
            border-radius: 12px;
            font-size: 1.2rem;
            font-weight: 700;
            color: #991b1b;
            margin-bottom: 0.8rem;
        }

        .result-ham {
            background: linear-gradient(90deg, #dcfce7, #bbf7d0);
            border-left: 6px solid #16a34a;
            padding: 1rem 1.2rem;
            border-radius: 12px;
            font-size: 1.2rem;
            font-weight: 700;
            color: #166534;
            margin-bottom: 0.8rem;
        }

        /* Confidence label */
        .confidence-label {
            font-size: 0.95rem;
            color: #4b5563;
            margin-top: 0.4rem;
            margin-bottom: 0.2rem;
        }

        /* Buttons */
        .stButton>button {
            width: 100%;
            background: linear-gradient(90deg, #4f46e5, #7c3aed);
            color: white;
            font-weight: 600;
            border: none;
            border-radius: 10px;
            padding: 0.6rem 0;
            font-size: 1.05rem;
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }
        .stButton>button:hover {
            transform: translateY(-1px);
            box-shadow: 0 6px 16px rgba(79,70,229,0.35);
            color: white;
        }

        /* Text area */
        .stTextArea textarea {
            border-radius: 12px;
            border: 1px solid #e0e3ea;
            font-size: 0.98rem;
        }

        /* Footer */
        .footer-text {
            text-align: center;
            color: #9ca3af;
            font-size: 0.85rem;
            margin-top: 1.5rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# ----------------------------------------------------------
# Load Trained Model and Vectorizer
# ----------------------------------------------------------

@st.cache_resource
def load_model():
    model = joblib.load("models/spam_model.pkl")
    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
    return model, vectorizer


try:
    model, vectorizer = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    load_error = str(e)


# ----------------------------------------------------------
# Sidebar
# ----------------------------------------------------------

with st.sidebar:
    st.markdown("## 📌 About This Project")
    st.markdown(
        """
        A machine learning system that classifies
        emails as **Spam** or **Ham (Normal)** using
        Natural Language Processing.
        """
    )

    st.markdown("---")
    st.markdown("### 🧠 Model")
    st.markdown("Linear Support Vector Machine (SVM)")

    st.markdown("### ⚙️ Techniques")
    st.markdown(
        """
        - Text preprocessing
        - TF-IDF Vectorization
        - Machine Learning Classification
        """
    )

    st.markdown("### 🛠️ Technologies")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("- Python\n- Scikit-Learn")
    with col2:
        st.markdown("- NLTK\n- Streamlit")

    st.markdown("---")
    st.caption("Made by Biruk Desalegn ")


# ----------------------------------------------------------
# Header
# ----------------------------------------------------------

st.markdown('<div class="main-title">📧 Spam Email Detector</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Paste an email below and let AI decide — Spam or Normal?</div>',
    unsafe_allow_html=True
)

if not model_loaded:
    st.error(f"⚠️ Could not load model files: {load_error}")
    st.stop()


# ----------------------------------------------------------
# Example Emails + Input
# ----------------------------------------------------------

st.markdown('<div class="card">', unsafe_allow_html=True)

st.markdown("####  Try an Example")
example_email = st.selectbox(
    "Choose a sample email",
    [
        "",
        "Congratulations! You won a free iPhone. Click here to claim your prize.",
        "Hey, are we meeting tomorrow at 3 PM?",
        "You have been selected for a cash reward. Call now!"
    ],
    label_visibility="collapsed"
)

st.markdown("#### ✍️ Email Content")
email_text = st.text_area(
    "Enter Email Content",
    value=example_email,
    height=200,
    placeholder="Paste or type the email message here...",
    label_visibility="collapsed"
)

analyze_clicked = st.button("🔍Analyze Email")

st.markdown('</div>', unsafe_allow_html=True)


# ----------------------------------------------------------
# Prediction Helpers
# ----------------------------------------------------------

def predict_email(email):
    cleaned_email = transform_text(email)
    vector = vectorizer.transform([cleaned_email])
    prediction = model.predict(vector)
    score = model.decision_function(vector)[0]
    return prediction[0], score, cleaned_email


def calculate_confidence(score):
    confidence = abs(score)
    percentage = min((confidence / 5) * 100, 99.9)
    return round(percentage, 2)


# ----------------------------------------------------------
# Run Prediction
# ----------------------------------------------------------

if analyze_clicked:

    if email_text.strip() == "":
        st.warning("⚠️ Please enter an email message.")

    else:
        with st.spinner("Analyzing email..."):
            time.sleep(0.4)
            prediction, score, cleaned_email = predict_email(email_text)
            confidence = calculate_confidence(score)

        st.markdown('<div class="card">', unsafe_allow_html=True)

        if prediction == 1:
            st.markdown(
                '<div class="result-spam">🚨 This email is SPAM</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="result-ham">✅ This email is NORMAL (HAM)</div>',
                unsafe_allow_html=True
            )

        st.markdown(f'<div class="confidence-label">Model Confidence: <b>{confidence}%</b></div>', unsafe_allow_html=True)
        st.progress(int(confidence))

        with st.expander("🔬 View Technical Details"):
            st.markdown("**Processed Email (after cleaning):**")
            st.code(cleaned_email, language="text")

            st.markdown("**Raw Decision Score:**")
            st.write(round(score, 3))

        st.markdown('</div>', unsafe_allow_html=True)


# ----------------------------------------------------------
# Footer
# ----------------------------------------------------------

st.divider()
st.markdown(
    '<div class="footer-text">Machine Learning Spam Detector &nbsp;|&nbsp; '
    'Built with Python, Scikit-Learn, NLP and Streamlit</div>',
    unsafe_allow_html=True
)