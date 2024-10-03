# Load the required libraries:
from sklearn import tree
from sklearn.metrics import accuracy_score
import numpy as np

#Import the Dataset
training_data = np.genfromtxt('dataset.csv', delimiter=',', dtype=np.int32) 
inputs = training_data[:,:-1]
outputs = training_data[:, -1]

#Divide dataset into training and testing data
training_inputs = inputs[:2000]
training_outputs = outputs[:2000]
testing_inputs = inputs[2000:]
testing_outputs = outputs[2000:]

#Use the Decision Tree Classifier Function (Algorithm)
classifier = tree.DecisionTreeClassifier()

#Train the model
classifier.fit(training_inputs, training_outputs)

#Compute the predictions
predictions = classifier.predict(testing_inputs)

#Calculate Accuracy of the model
accuracy = 100.0 * accuracy_score(testing_outputs, predictions)

#Print the result
print ("The accuracy of your decision tree on testing data is: " + str(accuracy))
