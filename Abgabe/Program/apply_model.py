import pandas as pd
import pickle as pk
import numpy as np
from sklearn.preprocessing import StandardScaler
import tkinter as tk
from tkinter import *
from tkinter import ttk, filedialog
from tkinter.filedialog import askopenfile
import os

print("Pick data:")
file = filedialog.askopenfile(mode='r', filetypes=[('CSV', '*.csv')])
if file:
    filepath = os.path.abspath(file.name)
print("The File is located at : " + str(filepath))

data = pd.read_csv(filepath)
# Delete unesessary features from dataset
data = data.dropna(how='all')
data = data.drop(['enterworld_num','buyitemnowmainauction_num','completechallengeweek_num'],axis=1)
data = data.drop(['survival_time'],axis=1)
data = data.drop(columns=data.columns[0],axis=1)
# Setup & split dataset for training
X = data.copy()
X = X.fillna(0)
account_ids = X.pop('actor_account_id')
s_scaler = StandardScaler()
X = s_scaler.fit_transform(X)


print("Pick model:")
sav = filedialog.askopenfile(mode='r', filetypes=[('Trained Model', '*.sav')])
if sav:
    sav = os.path.abspath(sav.name)
print("The File is located at : " + str(sav))

loaded_model = pk.load(open(sav, 'rb'))
predictions = loaded_model.predict(X)
np.unique(predictions, return_counts=True)

df_predictions = pd.DataFrame([account_ids,predictions]).T

df_predictions = df_predictions.rename(columns={'Unnamed 0':'churn_yn'})

df_predictions.to_csv('churn_predictions.csv')