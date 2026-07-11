# ==========================================================
# Spam Email Detection - Streamlit Application
# ==========================================================

import streamlit as st
import joblib
import sys


# ----------------------------------------------------------
# Import preprocessing function
# ----------------------------------------------------------

sys.path.append("src")

from preprocessing import transform_text



# ----------------------------------------------------------
# Load Trained Model and Vectorizer
# ----------------------------------------------------------

@st.cache_resource
def load_model():

    model = joblib.load(
        "models/spam_model.pkl"
    )

    vectorizer = joblib.load(
        "models/tfidf_vectorizer.pkl"
    )

    return model, vectorizer



model, vectorizer = load_model()



# ----------------------------------------------------------
# Page Configuration
# ----------------------------------------------------------

st.set_page_config(
    page_title="Spam Email Detector",
    page_icon="📧",
    layout="centered"
)



# ----------------------------------------------------------
# Sidebar
# ----------------------------------------------------------

st.sidebar.title(
    "📌 About This Project"
)


st.sidebar.write(
"""
## Spam Email Detection System

This application uses Machine Learning
and Natural Language Processing to classify
emails as Spam or Ham.

### Model
Linear Support Vector Machine (SVM)

### Techniques
- Text preprocessing
- TF-IDF Vectorization
- Machine Learning Classification

### Technologies
- Python
- Scikit-Learn
- NLTK
- Streamlit

"""
)



# ----------------------------------------------------------
# Title Section
# ----------------------------------------------------------

st.title(
    "📧 Spam Email Detection System"
)


st.write(
"""
Enter an email message below and the AI model
will predict whether it is Spam or Normal.
"""
)



# ----------------------------------------------------------
# Example Emails
# ----------------------------------------------------------

st.subheader(
    "Try an Example"
)


example_email = st.selectbox(
    "Choose an example email",
    [
        "",
        "Congratulations! You won a free iPhone. Click here to claim your prize.",
        "Hey, are we meeting tomorrow at 3 PM?",
        "You have been selected for a cash reward. Call now!"
    ]
)



# ----------------------------------------------------------
# Email Input
# ----------------------------------------------------------

email_text = st.text_area(
    "Enter Email Content",
    value=example_email,
    height=200
)



# ----------------------------------------------------------
# Prediction Function
# ----------------------------------------------------------

def predict_email(email):

    # Text preprocessing
    cleaned_email = transform_text(email)


    # Convert text into numbers
    vector = vectorizer.transform(
        [cleaned_email]
    )


    # Prediction
    prediction = model.predict(
        vector
    )


    # Confidence score
    score = model.decision_function(
        vector
    )[0]


    return prediction[0], score



# ----------------------------------------------------------
# Confidence Calculation
# ----------------------------------------------------------

def calculate_confidence(score):

    confidence = abs(score)


    percentage = min(
        (confidence / 5) * 100,
        99.9
    )


    return round(
        percentage,
        2
    )



# ----------------------------------------------------------
# Prediction Button
# ----------------------------------------------------------

if st.button(
    "🔍 Analyze Email"
):


    if email_text.strip() == "":

        st.warning(
            "Please enter an email message."
        )


    else:

        prediction, score = predict_email(
            email_text
        )


        confidence = calculate_confidence(
            score
        )



        # Spam result

        if prediction == 1:


            st.error(
                "🚨 This email is SPAM"
            )


        # Ham result

        else:


            st.success(
                "✅ This email is NORMAL (HAM)"
            )



        st.info(
            f"Model Confidence: {confidence}%"
        )



        # Debug information

        with st.expander(
            "View Details"
        ):

            st.write(
                "Processed Email:"
            )

            st.code(
                transform_text(email_text)
            )


            st.write(
                "Decision Score:"
            )

            st.write(
                round(score,3)
            )



# ----------------------------------------------------------
# Footer
# ----------------------------------------------------------

st.divider()


st.caption(
"""
Machine Learning Spam Detector |
Built with Python, Scikit-Learn, NLP and Streamlit
"""
)