import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score




def load_data():
    # Assuming the CSV is in the current working directory
    data = pd.read_csv('mail_data.csv')
    return data



def train_model():

    mail_data = load_data()

    # Label encoding: spam = 0, ham = 1
    mail_data['Category'] = mail_data['Category'].map({'spam': 0, 'ham': 1})


    X = mail_data['Message']
    y = mail_data['Category']


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)

    # Transform the text data to feature vectors
    vectorizer = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
    X_train_features = vectorizer.fit_transform(X_train)
    X_test_features = vectorizer.transform(X_test)


    model = LogisticRegression()
    model.fit(X_train_features, y_train)


    predictions = model.predict(X_test_features)
    accuracy = accuracy_score(y_test, predictions)

    return model, vectorizer, accuracy


# Main Streamlit app
def main():
    st.title("Spam Mail Prediction App")


    model, vectorizer, accuracy = train_model()

    st.write(f"Model trained with an accuracy of {accuracy * 100:.2f}%")


    input_message = st.text_area("Enter the email message")

    if st.button("Predict"):

        input_data_features = vectorizer.transform([input_message])


        prediction = model.predict(input_data_features)

        if prediction == 0:
            st.error("This email is classified as **Spam**.")
        else:
            st.success("This email is classified as **Ham** (Not Spam).")


if __name__ == '__main__':
    main()