import numpy as np
import pandas as pd
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load the data
df = pd.read_csv('data_url.csv', on_bad_lines='skip')
df = df.sample(n=10000, random_state=42)

# Select relevant columns
df = df[['label', 'url']]

# Remove null values
df = df.dropna()

# Create category_id
df['category_id'] = df['label'].factorize()[0]
category_id_df = df[['label', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'label']].values)

# Tokenizer function for URL by Faizan Ahmad, CEO FSecurify
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

# Prepare features and labels
vectorizer = TfidfVectorizer(tokenizer=getTokens, use_idf=True, smooth_idf=True, sublinear_tf=False)
features = vectorizer.fit_transform(df.url)
labels = df.label

# Split the data
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Train the model
clf = LogisticRegression(random_state=42)
clf.fit(X_train, y_train)

# Evaluate the model
train_score = clf.score(X_train, y_train)
test_score = clf.score(X_test, y_test)
print('Train accuracy =', train_score)
print('Test accuracy =', test_score)

# Prediction function
def predict_url(url):
    X_predict = vectorizer.transform([url])
    y_predict = clf.predict(X_predict)
    return y_predict[0]

# Example usage
while True:
    user_input = input("Enter URL (or 'quit' to exit): ")
    if user_input.lower() == 'quit':
        break
    prediction = predict_url(user_input)
    print(f"Prediction: {prediction}")
