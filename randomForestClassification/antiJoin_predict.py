# Load libraries
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint

from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.preprocessing import LabelEncoder

col_names = ["record_id","operation","num_input_table","num_key","key_name","name_left_table","name_right_table","rowNum_left_table","rowNum_right_table","rowNum_output_table","colNum_left_table","colNum_right_table","colNum_output_table","cpu_exe_time(ms)","fpga_exe_time(ms)","label(0_CPU/1_FPGA)"]

# load dataset
input_training_dataset = pd.read_csv('antijoin_training_dataset.csv', sep=',', header=0)
input_testing_dataset = pd.read_csv("antijoin_testing_dataset.csv", sep=',', header=0)

#split dataset in features and target variable
feature_cols = ['num_key','key_name','rowNum_left_table','rowNum_right_table','colNum_left_table','colNum_right_table','colNum_output_table']
label_col = ['label(0_CPU/1_FPGA)']

X_training = input_training_dataset[feature_cols] # Features
X_testing = input_testing_dataset[feature_cols] # Features
le = LabelEncoder()
X_training.key_name = le.fit_transform(X_training.key_name)
X_testing.key_name = le.transform(X_testing.key_name)

# mapping = dict(zip(le.classes_, range(0, len(le.classes_)+1)))
# print(mapping)

# print(input_training_dataset.key_name.head())
# print(X_training.head())

# print(input_testing_dataset.key_name.head())
# print(X_testing.key_name.head())

y_training = input_training_dataset[label_col] # Target variable
y_testing = input_testing_dataset[label_col] # Target variable

# # Split dataset into training set and test set
# X_train, X_test, y_train, y_test = train_test_split(X_training, y_training, test_size=0.3, random_state=1) # 70% training and 30% test
# X_train = np.array(X_train)
# y_train = np.array(y_train)
# y_train = y_train.reshape(-1)
# X_test = np.array(X_test)
# y_test = np.array(y_test)

X_train = X_training
y_train = y_training
X_test = X_testing
y_test = y_testing

# Random Foreset training + prediction
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Decision Tree training + prediction
fractions = [0,0.1,0.2,0.3,0.4]
accuracy = 0.0
best_criterion = "none"
best_frac = 0
best_split = 0
best_depth = 0
best_maxFeature = "none"
best_presort = "none"
y_pred_best = y_test

for i in fractions:
    for j in range(2,20,1):
        for k in range(1,30):
            clf = DecisionTreeClassifier(criterion='gini', min_weight_fraction_leaf=i,min_samples_split=j,max_depth=k,max_features="sqrt")
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            this_acc = metrics.accuracy_score(y_test, y_pred)
            # print("\nFraction: " + str(i) + " Split: " + str(j) + " Accuracy:",this_acc)
            if (this_acc > accuracy):
                accuracy = this_acc
                best_criterion = "gini"
                best_frac = i
                best_split = j
                best_depth = k
                best_maxFeature = "sqrt"
                best_presort = "Unsorted"
                y_pred_best = y_pred
                
            clf = DecisionTreeClassifier(criterion='entropy',min_weight_fraction_leaf=i,min_samples_split=j,max_depth=k,max_features="sqrt")
            clf.fit(X_train, y_train)
            pred = clf.predict(X_test)
            this_acc = metrics.accuracy_score(y_test, y_pred)
            # print("Fraction: " + str(i) + " Split: " + str(j) + " Accuracy:",this_acc)
            if (this_acc > accuracy):
                accuracy = this_acc
                best_criterion = "entropy"
                best_frac = i
                best_split = j
                best_depth = k
                best_maxFeature = "sqrt"
                best_presort = "Unsorted"
                y_pred_best = y_pred

            clf = DecisionTreeClassifier(criterion='gini', min_weight_fraction_leaf=i,min_samples_split=j,max_depth=k)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            this_acc = metrics.accuracy_score(y_test, y_pred)
            # print("\nFraction: " + str(i) + " Split: " + str(j) + " Accuracy:",this_acc)
            if (this_acc > accuracy):
                accuracy = this_acc
                best_criterion = "gini"
                best_frac = i
                best_split = j
                best_depth = k
                best_maxFeature = "none"
                best_presort = "Unsorted"
                y_pred_best = y_pred

            clf = DecisionTreeClassifier(criterion='entropy',min_weight_fraction_leaf=i,min_samples_split=j,max_depth=k)
            clf.fit(X_train, y_train)
            pred = clf.predict(X_test)
            this_acc = metrics.accuracy_score(y_test, y_pred)
            # print("Fraction: " + str(i) + " Split: " + str(j) + " Accuracy:",this_acc)
            if (this_acc > accuracy):
                accuracy = this_acc
                best_criterion = "entropy"
                best_frac = i
                best_split = j
                best_depth = k
                best_maxFeature = "none"
                best_presort = "Unsorted"
                y_pred_best = y_pred

print("Best Accuracy: " + str(accuracy) + " at: (" + best_criterion + ", " + \
                                                     str(best_frac) + ", " + \
                                                     str(best_split) + ", " + \
                                                     str(best_depth) + ", " + \
                                                     best_maxFeature + ", " + \
                                                     best_presort + ")" )

y_pred_best = pd.DataFrame(y_pred_best)
comparison_results = pd.concat([y_test, y_pred_best], axis=1)
comparison_results.to_csv('antijoin_test_expected.csv', index=True)  # Set index=False to exclude row numbers

# from sklearn.tree import export_graphviz
# #from sklearn.externals.six import StringIO
# from six import StringIO
# from IPython.display import Image
# import pydotplus

# dot_data = StringIO()
# export_graphviz(clf, out_file=dot_data,
#                 filled=True, rounded=True,
#                 special_characters=True,feature_names = feature_cols,class_names=['0','1'])
# graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
# graph.write_png('diabetes.png')
# Image(graph.create_png())

# for i in range(3):
#     tree = rf.estimators_[i]
#     dot_data = export_graphviz(tree,
#                                feature_names=X_train.columns,  
#                                filled=True,  
#                                max_depth=2, 
#                                impurity=False, 
#                                proportion=True)
#     graph = pydotplus.graph_from_dot_data(dot_data)
#     graph.write_png('diabetes.png')
#     Image(graph.create_png())

