import pandas as pd
kdd_df = pd.read_csv("kddcup_dataset.csv", index_col=None)
kdd_df.head()
kdd_df.shape
y = kdd_df["label"].values
from collections import Counter
Counter(y).most_common()
def label_anomalous(text):
    """Binarize target labels into normal or anomalous."""
    if text == "normal":
        return 0
    else:
        return 1
kdd_df["label"] = kdd_df["label"].apply(label_anomalous)
y = kdd_df["label"].values
counts = Counter(y).most_common()
contamination_parameter = counts[1][1] / (counts[0][1] + counts[1][1])
from sklearn.preprocessing import LabelEncoder
encodings_dictionary = dict()
for c in kdd_df.columns:
    if kdd_df[c].dtype == "object":
        encodings_dictionary[c] = LabelEncoder()
        kdd_df[c] = encodings_dictionary[c].fit_transform(kdd_df[c])
kdd_df_normal = kdd_df[kdd_df["label"] == 0]
kdd_df_abnormal = kdd_df[kdd_df["label"] == 1]
y_normal = kdd_df_normal.pop("label").values
X_normal = kdd_df_normal.values
y_anomaly = kdd_df_abnormal.pop("label").values
X_anomaly = kdd_df_abnormal.values
from sklearn.model_selection import train_test_split
X_normal_train, X_normal_test, y_normal_train, y_normal_test = train_test_split(
    X_normal, y_normal, test_size=0.3, random_state=11
)
X_anomaly_train, X_anomaly_test, y_anomaly_train, y_anomaly_test = train_test_split(
    X_anomaly, y_anomaly, test_size=0.3, random_state=11
)
import numpy as np
X_train = np.concatenate((X_normal_train, X_anomaly_train))
y_train = np.concatenate((y_normal_train, y_anomaly_train))
X_test = np.concatenate((X_normal_test, X_anomaly_test))
y_test = np.concatenate((y_normal_test, y_anomaly_test))
from sklearn.ensemble import IsolationForest
IF = IsolationForest(contamination=contamination_parameter)
IF.fit(X_train)
decisionScores_train_normal = IF.decision_function(X_normal_train)
decisionScores_train_anomaly = IF.decision_function(X_anomaly_train)
import matplotlib.pyplot as plt
plt.figure(figsize=(20, 10))
_ = plt.hist(decisionScores_train_normal, bins=50)
plt.show()
plt.figure(figsize=(20, 10))
_ = plt.hist(decisionScores_train_anomaly, bins=50)
plt.show()
cutoff = 0
print(Counter(y_test))
print(Counter(y_test[cutoff > IF.decision_function(X_test)]))
