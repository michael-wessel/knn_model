import pandas as pd
import numpy as np
import seaborn as sb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
import  matplotlib.pyplot as plt 
%matplotlib inline

# Read diabetes.csv from github and print its shape
url = 'https://raw.githubusercontent.com/susanli2016/Machine-Learning-with-Python/master/diabetes.csv'
diabetes = pd.read_csv(url, error_bad_lines=False)
print('Dimensions of diabetes data: {}'.format(diabetes.shape))

# Create a dataframe with all training data except the target column, in this case 'Outcome'
X = diabetes.drop(columns=['Outcome'])
X.head()

# Speaking of 'Outcomes', print and plot it
# 0=No Diabetes, 1=Diabetes
print(diabetes.groupby('Outcome').size())
sb.countplot(diabetes['Outcome'], label='Count')

# Seperate and view target values from the dataset
y = diabetes['Outcome'].values
y[0:5]

# Split data into training and test subsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

# Create the knn classifier
knn = KNeighborsClassifier(n_neighbors=3)

# Fit the classifier to the data, aka train the model
knn.fit(X_train, y_train)

# Predictions for test data based on fit model - firt 5
knn.predict(X_test)[0:5] # The model predicted that the first four patients had 'no diabets' and that the fifth had diabetes

# Test to see how accurate the model
knn.score(X_test, y_test) # Model has a 66.88% accuracy

# k-Fold Cross-Validation using a new knn model
knn_cv = KNeighborsClassifier(n_neighbors=3)

# Train the model with cross-validation of 5
scores_cv = cross_val_score(knn_cv, X, y, cv=5)

# Print the cross-validation scores and average score to depict the model's accuracy
print(scores_cv)
print('scores_cv mean: {}'.format(np.mean(scores_cv))) # Cross-validation model has a 71.36% accuracy

# Hypertuning model parameters using GridSearchCV
knn2 = KNeighborsClassifier()

# Create a dictionary of all values we want to tst for n_neighbors
parameter_grid = {'n_neighbors': np.arange(1, 25)}

# Use gridsearch to test values for n_neighbors
knn_gscv = GridSearchCV(knn2, parameter_grid, cv=5)

# Train the model with Gridsearch
knn_gscv.fit(X, y)

# Check top performaning n_neihbors value
knn_gscv.best_params_ # Optimal value = 14

# Check mean score for the top performing values of n_neighbors
knn_gscv.best_score_ # Model has a 75.78% accuracy

training_accuracy = []
test_accuracy = []

# Try neighbors from 1 to 10
neighbor_settings = range(1,15)

for n_neighbors in neighbor_settings:
    # Build the model
    knn_gscv2 = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_gscv2.fit(X_train, y_train)
    # Record training set accuracy
    training_accuracy.append(knn_gscv2.score(X_train, y_train))    
    # Record test set accuracy
    test_accuracy.append(knn_gscv2.score(X_test, y_test))

plt.plot(neighbor_settings, training_accuracy, label='Training Accuracy')
plt.plot(neighbor_settings, test_accuracy, label='Test Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('n_neighbors')
plt.legend()
plt.show()