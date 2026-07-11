import joblib

from preprocessing import transform_text


model = joblib.load(
    "../models/spam_model.pkl"
)

vectorizer = joblib.load(
    "../models/tfidf_vectorizer.pkl"
)


def predict_email(email):

    cleaned_email = transform_text(email)

    vector = vectorizer.transform(
        [cleaned_email]
    )

    prediction = model.predict(vector)

    probability = model.decision_function(vector)


    if prediction[0] == 1:
        result = "Spam"
    else:
        result = "Ham"


    return result, probability[0]



if __name__ == "__main__":

    email = """
    Congratulations!
    You won a free prize.
    Click now.
    """

    result, score = predict_email(email)

    print("Prediction:", result)
    print("Score:", score)