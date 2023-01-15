import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pickle
import tkinter as tk
from tkinter import *
from tkinter import ttk, filedialog
from tkinter.filedialog import askopenfile
import os
from time import sleep

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification
from numpy import asarray
from sklearn.preprocessing import scale

from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree



#tk.messagebox.showinfo(title=None, message="Pick model to evaluate")
print("Pick model to evaluate")
sleep(1)
file = filedialog.askopenfile(mode='r', filetypes=[('Pickle Model Files', '*.sav')])
if file:
    filepath_model = os.path.abspath(file.name)
print("The File is located at : " + str(filepath_model))

#tk.messagebox.showinfo(title=None, message="CSV/Table to evaluate")
print("CSV/Table to evaluate")
sleep(1)
file2 = filedialog.askopenfile(mode='r', filetypes=[('Pickle Model Files', '*.csv')])
if file2:
    filepath_csv = os.path.abspath(file2.name)
print("The File is located at : " + str(filepath_csv))



data = pd.read_csv(filepath_csv)
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

loaded_model = pickle.load(open(filepath_model, 'rb'))

#F1-score
print("Score for Selected model:", filepath_model)
print("F1 Score")
mean_score = cross_val_score(loaded_model, X, y, scoring="f1", cv = 10).mean()
std_score = cross_val_score(loaded_model, X, y, scoring="f1", cv = 10).std()
print('Mean F1:', mean_score)
print('Std F1 score', std_score)
#Cross validation
scores = cross_val_score(loaded_model, X, y, scoring="f1", cv=5)
print("CrossValidation Scores: ")
print(scores)
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))