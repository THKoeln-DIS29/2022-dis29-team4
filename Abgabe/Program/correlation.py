import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
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
#usecols=['churn_yn','longest_time_between_events', 'event_num', 'enterworld_num', 'levelup_num', 'spendmoney_num', 'itemupgrade_successrate', 'sessions_num'])
data = data.dropna()
data = data.drop(['actor_account_id','survival_time','churn_yn'],axis=1)
data = data.drop(columns=data.columns[0],axis=1)

corr = data.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
f, ax = plt.subplots(figsize=(14, 14))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
heat_map=sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": 0.8})
heat_map=heat_map.get_figure()
heat_map.savefig("output/correlation_map.png")
print("The Corelation map is saved to output/correlation_map.png")
