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
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import GridSearchCV


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree

import tkinter as tk
from tkinter import *
from tkinter import ttk, filedialog
from tkinter.filedialog import askopenfile
import os

def classy():

    s_scaler = StandardScaler()

    print("starting voting classifier script...")

    print("Pick data:")
    file = filedialog.askopenfile(mode='r', filetypes=[('CSV', '*.csv')])
    if file:
        filepath = os.path.abspath(file.name)
    print("The File is located at : " + str(filepath))
    print("loading features")
    data = pd.read_csv(filepath)
    # Delete unesessary features from dataset
    data = data.dropna()
    data = data.drop(['enterworld_num','buyitemnowmainauction_num','completechallengeweek_num'],axis=1)
    data = data.drop(['actor_account_id'
                        ,'survival_time'
                        ],axis=1)
    data = data.drop(columns=data.columns[0],axis=1)
    # Setup & split dataset for training
    print("building training & test dataset")
    X = data.copy()
    y = X.pop('churn_yn')
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 100)
    # Print training data shape
    print("Shape of test and training dataset")
    print(X_test.shape, y_test.shape)
    print(X_train.shape, y_train.shape)

    print("normalizing dataset")
    X_train = s_scaler.fit_transform(X_train)
    X_test = s_scaler.fit_transform(X_test)

    print("Building voting classifier")

    print("seting up decision tree")
    clf = DecisionTreeClassifier(criterion='entropy', 
        splitter='best', 
        max_depth=6, 
        min_samples_split=2, 
        min_samples_leaf=4, 
        min_weight_fraction_leaf=0.0, 
        max_features='sqrt', 
        random_state=None, 
        max_leaf_nodes=None, 
        min_impurity_decrease=0.0, 
        ccp_alpha=0.0)

    print("seting up random forrest")
    forest = RandomForestClassifier(random_state=0,
                                criterion='entropy',
                                max_depth=6,
                                max_features='sqrt',
                                n_estimators=1000)

    print("seting up gaussian process classifier")
    gapc=GaussianProcessClassifier(1.0 * RBF(1.0), 
                                    multi_class=('one_vs_one'),
                                    max_iter_predict=100,
                                    n_restarts_optimizer=5,
                                    n_jobs=-1)

    print("combining classifiers")
    eclf = VotingClassifier(estimators=[('RF', forest),
                                        ('Gaussian Process', gapc),
                                        ('DCTree', clf)],
                                        voting='soft', weights=[1,1,1])

    print("training voting classifier")
    eclf.fit(X_train, y_train)

    #F1-score
    print("F1 Score")
    vote = cross_val_score(eclf, X, y, scoring="f1", cv = 5,n_jobs=-1)
    print("CrossValidation Scores: ")
    print(vote)
    print("%0.2f accuracy with a standard deviation of %0.2f" % (vote.mean(), vote.std()))

    filename = 'output/voting_clf.sav'
    pickle.dump(eclf, open(filename, 'wb'))
    print("The model was trained and saved to output/voting_clf.sav")