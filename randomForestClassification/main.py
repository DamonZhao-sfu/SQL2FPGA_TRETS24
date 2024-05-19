# Load libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
# load dataset
pima = pd.read_csv("diabetes.csv", header=0, names=col_names)

# print(pima.head())

#split dataset in features and target variable
feature_cols = ['insulin', 'bmi', 'age','glucose','bp','pedigree']
X = pima[feature_cols] # Features
y = pima.label # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

print(X_train.head())
# print(y_train.head())

# Create Decision Tree classifer object
fractions = [0,0.1,0.2,0.3,0.4]

accuracy = 0.0
best_criterion = "none"
best_frac = 0
best_split = 0
best_depth = 0
best_maxFeature = "none"
best_presort = "none"

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

print("Best Accuracy: " + str(accuracy) + " at: (" + best_criterion + ", " + \
                                                     str(best_frac) + ", " + \
                                                     str(best_split) + ", " + \
                                                     str(best_depth) + ", " + \
                                                     best_maxFeature + ", " + \
                                                     best_presort + ")" )

from sklearn.tree import export_graphviz
#from sklearn.externals.six import StringIO
from six import StringIO
from IPython.display import Image
import pydotplus

dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True,feature_names = feature_cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('diabetes.png')
Image(graph.create_png())

