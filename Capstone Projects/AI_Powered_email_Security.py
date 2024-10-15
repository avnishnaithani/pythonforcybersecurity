import numpy as np
import pandas as pd
import re
import nltk
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

def train_spam_model():
    print("Training spam detection model...")
    df = pd.read_csv("spam.csv", encoding="latin-1")
    df = df.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
    df = df.rename(columns={"v1": "label", "v2": "text"})

    # Preprocess text data
    df["text"] = df["text"].str.lower()
    df["text"] = df["text"].apply(word_tokenize)
    stop_words = set(stopwords.words("english"))
    df["text"] = df["text"].apply(lambda x: [word for word in x if word not in stop_words])
    df["text"] = df["text"].apply(lambda x: " ".join(x))

    # Feature Extraction
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df["text"])

    # Split the Dataset
    X_train, X_test, y_train, y_test = train_test_split(X, df["label"], test_size=0.2, random_state=42)

    # Train a classification model
    classifier = MultinomialNB()
    classifier.fit(X_train, y_train)

    # Evaluate the model's performance
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print("Accuracy:", accuracy)
    print("Classification Report:")
    print(report)

    # Save the model and vectorizer
    joblib.dump(classifier, 'email_spam_model.pkl')
    joblib.dump(vectorizer, 'email_spam_vectorizer.pkl')
    print("Spam detection model trained and saved.")

def check_spam():
    try:
        classifier = joblib.load('email_spam_model.pkl')
        vectorizer = joblib.load('email_spam_vectorizer.pkl')
    except FileNotFoundError:
        print("Model not found. Training a new model...")
        train_spam_model()
        classifier = joblib.load('email_spam_model.pkl')
        vectorizer = joblib.load('email_spam_vectorizer.pkl')

    new_email = [input("Enter the email text: ")]
    new_email = vectorizer.transform(new_email)
    prediction = classifier.predict(new_email)
    if prediction[0] == "spam":
        print("This email is likely spam.")
    else:
        print("This email is likely not spam.")

def getTokens(input):
    tokensBySlash = str(input).split('/')
    allTokens = []
    for i in tokensBySlash:
        tokens = str(i).split('-')
        tokensByDot = []
        for j in range(0, len(tokens)):
            tempTokens = str(tokens[j]).split('.')
            tokensByDot = tokensByDot + tempTokens
        allTokens = allTokens + tokens + tokensByDot
    allTokens = list(set(allTokens))
    if 'com' in allTokens:
        allTokens.remove('com')
    return allTokens

def train_url_model():
    print("Training URL detection model...")
    df = pd.read_csv('data_url.csv', on_bad_lines='skip')
    df = df.sample(n=10000, random_state=42)
    df = df[['label', 'url']]
    df = df.dropna()
    df['category_id'] = df['label'].factorize()[0]

    vectorizer = TfidfVectorizer(tokenizer=getTokens, use_idf=True, smooth_idf=True, sublinear_tf=False)
    features = vectorizer.fit_transform(df.url)
    labels = df.label

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    clf = LogisticRegression(random_state=42)
    clf.fit(X_train, y_train)

    train_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)
    print('Train accuracy =', train_score)
    print('Test accuracy =', test_score)

    joblib.dump(clf, 'url_detection_model.pkl')
    joblib.dump(vectorizer, 'url_detection_vectorizer.pkl')
    print("URL detection model trained and saved.")

def check_url():
    try:
        clf = joblib.load('url_detection_model.pkl')
        vectorizer = joblib.load('url_detection_vectorizer.pkl')
    except FileNotFoundError:
        print("Model not found. Training a new model...")
        train_url_model()
        clf = joblib.load('url_detection_model.pkl')
        vectorizer = joblib.load('url_detection_vectorizer.pkl')

    url = input("Enter URL: ")
    X_predict = vectorizer.transform([url])
    y_predict = clf.predict(X_predict)
    print(f"Prediction: {y_predict[0]}")

def main_menu():
    while True:
        print("\nAI-powered Email Security Menu:")
        print("1. Check for spam/phishing text")
        print("2. Check for malicious URLs")
        print("3. Exit")
        
        choice = input("Enter your choice (1-3): ")
        
        if choice == '1':
            check_spam()
        elif choice == '2':
            check_url()
        elif choice == '3':
            print("Exiting the program. Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main_menu()
