# Importing necessary libraries
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.svm import SVC

# Generating random data and visualizing it using Matplotlib
ys = 200 + np.random.randn(100)
x = [x for x in range(len(ys))]

plt.plot(x, ys, '-')
plt.fill_between(x, ys, 195, where=(ys > 195), facecolor='g', alpha=0.6)
plt.title("Sample Visualization")
plt.show()

# Loading diabetes dataset and exploring its basic information
df = pd.read_csv("/content/drive/MyDrive/diabetes.csv")
print(df)
print(df.info())
print(df.describe())

# Handling missing values by filling them with mean
df.fillna(df.mean(), inplace=True)

# Exploring the distribution of the target variable 'Outcome' using Seaborn
print(df.Outcome.value_counts())
sns.countplot(df['Outcome'])
plt.show()

# Exploring the distribution of the 'Pregnancies' column using Seaborn
sns.distplot(df["Pregnancies"])
plt.show()

# Visualizing histograms for all columns in the dataset
p = df.hist(figsize=(10, 10))

# Highlighting max and min values in the first 10 rows of the dataset
df.head(10).style.highlight_max(color="lightblue").highlight_min(color="red")

# Visualizing pair plots and correlation heatmap using Seaborn
graph = ['Glucose', 'Insulin', 'BMI', 'Age', 'Outcome']
sns.set()
print(sns.pairplot(df[graph], hue='Outcome', diag_kind='kde'))
plt.figure(figsize=(13, 8))
sns.heatmap(df.corr(), annot=True, cmap='terrain')
plt.show()

# Splitting the data into features (X) and target variable (Y)
X = df.drop('Outcome', axis=1)
Y = df['Outcome']

# Splitting the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=4, stratify=Y)
print('TrainSet', X_train.shape)
print('TestSet', X_test.shape)

# Building a K-Nearest Neighbors (KNN) classifier and performing grid search for hyperparameter tuning
model = KNeighborsClassifier(n_neighbors=28, weights='distance')
param_grid = {'n_neighbors': np.arange(1, 51), 'metric': ['minkowski', 'manhattan']}
grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=4)
grid.fit(X, Y)
print(grid.best_score_)
print(grid.best_params_)
print(model.score(X_train, Y_train))
print(model.score(X_test, Y_test))

# Calculating and visualizing confusion matrix for the training set
Y_train_pred = model.predict(X_train)
CM_train = confusion_matrix(Y_train, Y_train_pred)
print("Confusion matrix: \n", CM_train)
print("Train Score ", model.score(X_train, Y_train))
sns.heatmap(CM_train, square=True, annot=True, cbar=False)

# Calculating accuracy score using cross-validation and on the test set
Acc_knn = cross_val_score(KNeighborsClassifier(), X, Y, cv=4, scoring='accuracy')
print(Acc_knn)
pred = model.predict(X_test)
prediction = model.predict(X_test)
print(f"Accuracy Score = {accuracy_score(Y_test, pred)}")

# Building a Support Vector Machine (SVM) classifier and performing grid search for hyperparameter tuning
classifier = SVC(kernel='linear', random_state=0)
classifier.fit(X_train, Y_train)
y_pred = classifier.predict(X_test)
print(y_pred)
cm = confusion_matrix(Y_test, y_pred)
print(cm)
sns.heatmap(cm, square=True, annot=True, cbar=False)

param_grid = {'C': [0.1, 1, 10], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 'kernel': ['rbf', 'linear']}
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3)
grid.fit(X_train, Y_train)
print(grid.best_params_)
print(grid.best_estimator_)
accuracy_score_svm = accuracy_score(Y_test, y_pred)
print(accuracy_score_svm)
