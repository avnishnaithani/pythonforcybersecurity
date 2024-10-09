import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier

# Load and shuffle the dataset
df = pd.read_csv("passwordDataset.csv", dtype={"password": "str", "strength": "int"}, index_col=None)
df = df.sample(frac=1)

# Split into training and testing sets
l = len(df.index)
train_df = df.head(int(l * 0.8))
test_df = df.tail(int(l * 0.2))

# Separate labels and features
y_train = train_df.pop("strength").values
y_test = test_df.pop("strength").values
X_train = train_df.values.flatten()
X_test = test_df.values.flatten()

# Custom tokenizer function to split passwords into characters
def character_tokens(input_string):
    return [x for x in input_string]

# Build the pipeline with TfidfVectorizer and XGBClassifier
password_clf = Pipeline([
    ("vect", TfidfVectorizer(tokenizer=character_tokens, token_pattern=None)),  # Set token_pattern None
    ("clf", XGBClassifier())
])

# Train the classifier
password_clf.fit(X_train, y_train)

# Evaluate the classifier
score = password_clf.score(X_test, y_test)
print(f"Model accuracy: {score}")

# Test predictions
entered_password = input("Enter password to determine its strength:")
predictions = password_clf.predict([entered_password])
print(f"Predictions: {predictions}")

if predictions==[0]:
    print("Password Strength: Weak")
elif predictions==[1]:
    print("Password Strength: Medium")
elif predictions==[2]:
    print("Password Strength: Strong")
