import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix

# Load and preprocess data
pwd_data = pd.read_csv("DSL-StrongPasswordData.csv", header=0)

# Average Keystroke Latency per Subject
DD = [dd for dd in pwd_data.columns if dd.startswith('DD')]
plot = pwd_data[DD]
plot['subject'] = pwd_data['subject'].values
plot = plot.groupby('subject').mean()
plot.iloc[:6].T.plot(figsize=(8, 6), title='Average Keystroke Latency per Subject')
plt.show()

# Split data
data_train, data_test = train_test_split(pwd_data, test_size=0.2, random_state=0)
X_train = data_train[pwd_data.columns[2:]]
y_train = data_train['subject']
X_test = data_test[pwd_data.columns[2:]]
y_test = data_test['subject']

# K-Nearest Neighbor Classifier
knc = KNeighborsClassifier()
knc.fit(X_train, y_train)
knc_pred = knc.predict(X_test)
knc_accuracy = metrics.accuracy_score(y_test, knc_pred)
print('K-Nearest Neighbor Classifier Accuracy:', knc_accuracy)

# Support Vector Classifier
svc = svm.SVC(kernel='linear') 
svc.fit(X_train, y_train)
svc_pred = svc.predict(X_test)
svc_accuracy = metrics.accuracy_score(y_test, svc_pred)
print('Support Vector Linear Classifier Accuracy:', svc_accuracy)

# Multi Layer Perceptron Classifier
mlpc = MLPClassifier()
mlpc.fit(X_train, y_train)
mlpc_pred = mlpc.predict(X_test)
mlpc_accuracy = metrics.accuracy_score(y_test, mlpc_pred)
print('Multi Layer Perceptron Classifier Accuracy:', mlpc_accuracy)

# Confusion Matrix
labels = sorted(pwd_data['subject'].unique())
cm = confusion_matrix(y_test, mlpc_pred, labels=labels)

# Plot Confusion Matrix
plt.figure(figsize=(10, 8))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix for MLP Classifier')
plt.colorbar()
tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels, rotation=90)
plt.yticks(tick_marks, labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.show()
