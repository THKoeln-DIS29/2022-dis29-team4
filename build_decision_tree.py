import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV



from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

data = pd.read_csv('features.csv')
# Delete unesessary features from dataset
data = data.dropna()
data = data.drop(['enterworld_num','buyitemnowmainauction_num','completechallengeweek_num'],axis=1)
data = data.drop(['actor_account_id'
                    ,'survival_time'
                    ],axis=1)
data = data.drop(columns=data.columns[0],axis=1)
# Setup & split dataset for training
X = data.copy()
y = X.pop('churn_yn')
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 100)
# Print training data shape
print(X_test.shape, y_test.shape)
print(X_train.shape, y_train.shape)
# Get feature names from training set
feature_names = [f"feature {i}" for i in range(X.shape[1])]

clf = DecisionTreeClassifier(criterion='gini',
    max_depth=4,
    splitter='best',
    min_samples_split=2, 
    min_samples_leaf=4,
    max_features='sqrt'
    )
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

#F1-score
print("F1 Score")
print('f1 ' , f1_score(y_test, predictions))
#Cross validation
scores = cross_val_score(clf, X, y, cv=5)
print("CrossValidation Scores: ")
print(scores)
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

filename = 'tree_model_test.sav'
pickle.dump(clf, open(filename, 'wb'))