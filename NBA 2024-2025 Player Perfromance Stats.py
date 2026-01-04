# -*- coding: utf-8 -*-
"""
Created on Mon Dec 15 16:32:54 2025

@author: Micah
"""

import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from scipy import stats
import os

os.getcwd()

nba_stats = pd.read_excel(
    "normal_stats.xlsx")

advanced = pd.read_csv(
    "advanced_files.csv"
)


def clean_col(col):
    col = col.strip()
    col = col.replace("Data", "date")
    col = col.replace(" ", "_")
    col = col.replace("(", "")
    col = col.replace(")", "")
    col = col.lower()
    return col

new_columns = []

for c in nba_stats.columns:
    clean_c = clean_col(c)
    new_columns.append(clean_c)
    
nba_stats.columns = new_columns
    

