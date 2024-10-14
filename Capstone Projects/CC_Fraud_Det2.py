# Data preparation and configuration setup

import numpy as np   # Import the NumPy library, which provides support for arrays, matrices, and mathematical functions in Python.
import pandas as pd  # Import the Pandas library, which provides support for data manipulation and analysis in Python.
import scipy         # Import the SciPy library, which provides support for scientific and technical computing in Python.
import warnings      # Import the warnings library, which allows you to filter out warnings that may be displayed during the execution of your code
LABELS = ["Normal", "Fraud"]        # Define the labels for the two classes.
np.random.seed(7)                   # Set random seed for reproducibility
warnings.filterwarnings('ignore')   # Ignore any warnings that may be displayed.

# Reading CSV file into dataframe
pd.set_option('display.max_columns',None) # This code sets the maximum number of columns to display in a Pandas dataframe to None, which ensures that all columns will be visible when the dataframe is printed.
# pd.set_option('display.max_rows',None)
df = pd.read_csv('creditcard.csv',sep=',') # sep=',' tells Pandas to split the file into columns wherever there is a comma.

# Displays first 5 rows
df.head()

# Displays last 5 rows
df.tail()

# Returns the dataframe shape
df.shape

# Displays information about the dataframe
df.info()

# Data types of dataframe columns
df.dtypes

# Checking for missing values
df.isnull().sum(axis = 0)

# Generates descriptive statistics for the dataframe
df.describe()

# Statistical data visualization
import matplotlib.pyplot as plt # For creating visualizations in Python
import seaborn as sns # For creating statistical visualizations in Python
from pylab import rcParams # rcParams module from the PyLab library, which allows you to customize the properties of your visualizations
rcParams['figure.figsize'] = 14, 8 # Figures created by Pyplot to 14 inches by 8 inches

# Histogram of the Credit Card Dataset

fig = plt.figure(figsize = (50,40))
df.hist(ax = fig.gca());

# Creating a correlation heatmap
sns.heatmap(df.corr(),annot=True, cmap='Spectral', linewidths=0.1)
fig=plt.gcf()
fig.set_size_inches(20,20)
plt.show()

# Transaction Class Distribution
count_classes = pd.value_counts(df['Class'], sort = True) # Counts the number of occurrences of each class in the 'Class' column of a Pandas dataframe and assigns the result to the variable count_classes
count_classes.plot(kind = 'bar', rot=0) # Creates a bar chart of the class distribution using the Matplotlib library, with the x-axis representing the class labels and the y-axis representing the frequency of each class.
plt.title("Transaction Class Distribution")
plt.xticks(range(2), LABELS) # Sets the x-axis tick labels to "Normal" and "Fraud" using the LABELS list defined earlier.
plt.xlabel("Class")
plt.ylabel("Frequency")

# Get the Fraud and the normal dataset 
fraud = df[df['Class']==1]  # Create a new dataframe called 'fraud' that contains only the rows from the original dataframe where the 'Class' column is equal to 1 (indicating a fraudulent transaction).
normal = df[df['Class']==0] # Create normal dataframe from original.

print(fraud.shape,normal.shape)

# Calculate class distribution percentage
df.Class.value_counts(normalize=True)*100

# We need to analyze more amount of information from the transaction data
# How different are the amount of money used in different transaction classes?
fraud.Amount.describe()

normal.Amount.describe()

f, (ax1, ax2) = plt.subplots(2, 1, sharex=True) # figure with two subplots arranged vertically and shares the x-axis between them.
f.suptitle('Amount per transaction by class') # Title of the figure.
bins = 50  # Bin is a range of values that are grouped together to form a bar in the histogram.
ax1.hist(fraud.Amount, bins = bins) # Create a histogram of the 'Amount' column in the 'fraud' subset of the data, with the specified number of bins.
ax1.set_title('Fraud') # Title of the first subplot
ax2.hist(normal.Amount, bins = bins) # Create a histogram of the 'Amount' column in the 'normal' subset of the data, with the specified number of bins.
ax2.set_title('Normal') # Title of the second subplot.
plt.xlabel('Amount ($)') # The x-axis label for the figure
plt.ylabel('Number of Transactions') # The y-axis label for the figure
plt.xlim((0, 20000)) # Limits of the x-axis to be from 0 to 20,000.
plt.yscale('log') # Y-axis scale to be logarithmic.
plt.show(); # Displays the figure

# We Will check Do fraudulent transactions occur more often during certain time frame ? Let us find out with a visual representation.

f, (ax1, ax2) = plt.subplots(2, 1, sharex=True) #  Create a new figure and two subplots with a shared x-axis.
f.suptitle('Time of transaction vs Amount by class') # Title to the entire figure.
ax1.scatter(fraud.Time, fraud.Amount) # Create a scatter plot of time vs amount for fraudulent transactions on the first subplot.
ax1.set_title('fraud') # Title for the first subplot
ax2.scatter(normal.Time, normal.Amount) # Create a scatter plot of time vs amount for non-fraudulent transactions on the second subplot.
ax2.set_title('normal') # Title for the second subplot.
plt.xlabel('Time (in Seconds)') # Set the x-axis label for the entire figure.
plt.ylabel('Amount') # Set the y-axis label for the entire figure.
plt.show() # Display the entire figure.

# Taking some sample data from population data

df1= df.sample(frac = 0.1,random_state=1)
df1.shape

df.shape

# Determine the number of fraud and valid transactions in the dataset

Fraud = df1[df1['Class']==1]
Valid = df1[df1['Class']==0]

outlier_fraction = len(Fraud)/float(len(Valid))

print(outlier_fraction)
print("Fraud Cases : {}".format(len(Fraud))) # Fraudulent transactions in the dataset, which is determined by counting the number of entries in the Fraud list.
print("Valid Cases : {}".format(len(Valid))) # show the number of valid (non-fraudulent) transactions in the dataset, which is determined by counting the number of entries in the Valid list.

# Create independent and Dependent Features
columns = df1.columns.tolist() #  Creates a list of all the columns in the dataframe df1 and stores it in the variable columns.

# Filter the columns to remove data we do not want 
columns = [c for c in columns if c not in ["Class"]] # filters the list of columns to remove the column named "Class", since this is the variable we are trying to predict and we do not want it included as a feature.

# Store the variable we are predicting 
target = "Class" # The column we are trying to predict in the variable target.

# Define a random state 
state = np.random.RandomState(42) # Create a random state object using NumPy's random.RandomState() method with a seed of 42. This is used to ensure that we can reproduce the same random numbers each time we run the code.
X = df1[columns] # Create a new dataframe X that includes only the columns we want to use as features (i.e., all columns except the "Class" column).
y = df1[target] # Create a new series Y that includes only the values from the "Class" column, which is the variable we are trying to predict.
X_outliers = state.uniform(low=0, high=1, size=(X.shape[0], X.shape[1]))

# Print the shapes of X and y
print(X.shape) 
print(y.shape)

import sklearn # Machine learning algorithm
from sklearn.ensemble import IsolationForest # Isolation Forest algorithm from Scikit-learn, which is an unsupervised learning method for anomaly detection.
from sklearn.neighbors import LocalOutlierFactor # Local Outlier Factor algorithm from Scikit-learn, which is a density-based method for anomaly detection.
from sklearn.svm import OneClassSVM # One-Class Support Vector Machine algorithm from Scikit-learn, which is a binary classification method for anomaly detection.
from sklearn.metrics import classification_report,accuracy_score # Evaluate the performance of a machine learning model.

# Define the outlier detection methods

classifiers = {
    "Isolation Forest":IsolationForest(n_estimators=100, max_samples=len(X), 
                                       contamination=outlier_fraction,random_state=state, verbose=0),
    "Local Outlier Factor":LocalOutlierFactor(n_neighbors=20, algorithm='auto', 
                                              leaf_size=30, metric='minkowski',
                                              p=2, metric_params=None, contamination=outlier_fraction),
    "Support Vector Machine":OneClassSVM(kernel='rbf', degree=3, gamma=0.1,nu=0.05, max_iter=-1)
}

type(classifiers) # Data type of the classifiers

n_outliers = len(Fraud)  # Calculate number of outliers (fraudulent transactions).
for i, (clf_name,clf) in enumerate(classifiers.items()): # Iterate through classifiers and their names.

    # Fit the data and tag outliers
    if clf_name == "Local Outlier Factor": # Check if using Local Outlier Factor
        y_pred = clf.fit_predict(X)        # Fit data and predict outliers.
        scores_prediction = clf.negative_outlier_factor_ # Get outlier scores
    elif clf_name == "Support Vector Machine":  # Check if using Support Vector Machine.
        clf.fit(X)                              # Fit/Learn data
        y_pred = clf.predict(X)                 # Predict outliers
    else:
        clf.fit(X)                              # For all other classifiers
        scores_prediction = clf.decision_function(X) # Get decision function score
        y_pred = clf.predict(X)                      # Predict outliers

    # Reshape the prediction values to 0 for Valid transactions , 1 for Fraud transactions
    y_pred[y_pred == 1] = 0        # Set predicted labels to 0 for valid transactions.
    y_pred[y_pred == -1] = 1       # Set predicted labels to 1 for fraudulent transactions
    n_errors = (y_pred != y).sum() # Count errors (mislabeled transactions).

    # Run Classification Metrics
    print("{}: {}".format(clf_name, n_errors))                              # Print classifier name and number of errors.
    print("Accuracy Score :", accuracy_score(y, y_pred), end=" ")           # Print accuracy score.
    print("\nClassification Report :\n", classification_report(y, y_pred))  # Print classification report.

