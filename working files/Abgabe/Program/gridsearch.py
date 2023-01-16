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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from numpy import asarray

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree


import tkinter as tk
from tkinter import *
from tkinter import ttk, filedialog
from tkinter.filedialog import askopenfile
import os

def grid():

    s_scaler = StandardScaler()

    print("starting gridsearch script...")

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
    X = data.copy()
    y = X.pop('churn_yn')
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 100)
    # Normalize data

    print("normalizing dataset")
    X_train = s_scaler.fit_transform(X_train)
    X_test = s_scaler.fit_transform(X_test)

    # Print training data shape
    print("Shape of test and training dataset")
    print(X_test.shape, y_test.shape)
    print(X_train.shape, y_train.shape)


    param_grid = { 
        'max_features': ['sqrt','log2'],
        'max_depth' : range(4,25),
        'criterion' :['gini','entropy'],
        'min_samples_split' : range (2,10),
        'min_samples_leaf' : range (2,5),
        'min_weight_fraction_leaf' : (0.0,0.1,0.2,0.3)
    }

    print("Decision Tree CV with param grid: " + str(param_grid))
    tree = DecisionTreeClassifier()
    CV_rfc = GridSearchCV(estimator=tree,param_grid=param_grid, cv= 5)
    CV_rfc.fit(X_train, y_train)

    print("best parameters for Decision Tree Classifier are:")
    print(CV_rfc.best_params_)
    print("best estimators for Decision Tree Classifier are:")
    print(CV_rfc.best_estimator_)

    param_grid = { 
        'n_estimators': range(900, 1200,100),
        'max_features': ['sqrt','log2'],
        'max_depth' : range(5,20,5),
        'criterion' :['gini','entropy']
    }
    print("Random Forrest CV with param grid: " + str(param_grid))
    forest = RandomForestClassifier(random_state=0)
    CV_rfc = GridSearchCV(estimator=forest,param_grid=param_grid, cv= 5)
    CV_rfc.fit(X_train, y_train)
    print("best parameters for Random Forrest are:")
    print(CV_rfc.best_params_)
