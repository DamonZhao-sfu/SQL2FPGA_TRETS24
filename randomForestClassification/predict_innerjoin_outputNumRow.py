# Load libraries
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt  
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.linear_model import LinearRegression # Linear Regression
from sklearn import svm # non-linear regression
from sklearn.tree import DecisionTreeRegressor
import random

col_names = ['record_id,operation','num_input_table','num_key','num_unique_key','key_name','name_left_table','name_right_table','rowNum_left_table','rowNum_right_table','leftNumRowLE1000','leftNumRowLE100000','leftNumRowLE1000000','rightNumRowLE1000','rightNumRowLE100000','rightNumRowLE1000000','rowNum_output_table','colNum_left_table','colNum_right_table','colNum_output_table','cpu_exe_time(ms)','fpga_exe_time(ms)','label(0_CPU/1_FPGA)']
# load dataset
input_dataset = pd.read_csv("innerjoin_training_dataset.csv", header=1, names=col_names)

#split dataset in features and target variable
feature_cols = ['num_unique_key', 'key_name', 'name_left_table','name_right_table','rowNum_left_table','rowNum_right_table','leftNumRowLE1000','leftNumRowLE100000','leftNumRowLE1000000','rightNumRowLE1000','rightNumRowLE100000','rightNumRowLE1000000']
label_col = ['rowNum_output_table']
X = input_dataset[feature_cols] # Features
le = LabelEncoder()
X.key_name = le.fit_transform(X.key_name)
X.name_left_table = le.fit_transform(X.name_left_table)
X.name_right_table = le.fit_transform(X.name_right_table)
y = input_dataset[label_col] # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=1) # 70% training and 30% test

# print(X_train.head())
# print(y_train.head())

X_train = np.array(X_train)
y_train = np.array(y_train)
y_train = y_train.reshape(-1)
X_test = np.array(X_test)
y_test = np.array(y_test)

# Linear Regression training + prediction
regressor = LinearRegression()  
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
print('Linear Regression')
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('')

# SVM training + prediction
regr = svm.SVR(kernel='poly')
regr.fit(X_train, y_train)
y_pred = regr.predict(X_test)
print('Support Vector Machine')
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('')

# Decision Tree training + prediction
regr = DecisionTreeRegressor(max_depth=20)
regr.fit(X_train, y_train)
y_pred = regr.predict(X_test)
print('Decision Tree')
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('')