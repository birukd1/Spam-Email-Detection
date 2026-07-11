import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from preprocessing import transform_text



# -------------------------
# Load Dataset
# -------------------------

def load_data():

    df = pd.read_csv(
        "../data/spam_clean.csv"
    )

    return df



# -------------------------
# Feature Engineering
# -------------------------

def create_features(df):

    tfidf = TfidfVectorizer(
        max_features=3000
    )


    X = tfidf.fit_transform(
        df["transformed_message"]
    )


    y = df["label"]


    return X, y, tfidf



# -------------------------
# Train Model
# -------------------------

def train_model(X_train, y_train):

    model = LinearSVC(
        C=1,
        random_state=42
    )


    model.fit(
        X_train,
        y_train
    )


    return model



# -------------------------
# Evaluation
# -------------------------

def evaluate(model, X_test, y_test):

    prediction = model.predict(
        X_test
    )


    print(
        "Accuracy:",
        accuracy_score(
            y_test,
            prediction
        )
    )


    print(
        "Precision:",
        precision_score(
            y_test,
            prediction
        )
    )


    print(
        "Recall:",
        recall_score(
            y_test,
            prediction
        )
    )


    print(
        "F1 Score:",
        f1_score(
            y_test,
            prediction
        )
    )



# -------------------------
# Save Model
# -------------------------

def save_model(model, vectorizer):

    joblib.dump(
        model,
        "../models/spam_model.pkl"
    )


    joblib.dump(
        vectorizer,
        "../models/tfidf_vectorizer.pkl"
    )


    print("Model saved successfully!")



# -------------------------
# Main Pipeline
# -------------------------

def main():

    print("Loading dataset...")

    df = load_data()


    print("Preprocessing text...")


    df["transformed_message"] = (
        df["message"]
        .apply(transform_text)
    )


    print("Creating features...")


    X, y, vectorizer = create_features(df)



    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


    print("Training model...")


    model = train_model(
        X_train,
        y_train
    )


    print("Evaluating model...")


    evaluate(
        model,
        X_test,
        y_test
    )


    save_model(
        model,
        vectorizer
    )



if __name__ == "__main__":

    main()