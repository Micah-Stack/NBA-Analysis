# -*- coding: utf-8 -*-
"""
Created on Mon Dec 15 16:32:54 2025

@author: Micah
"""

import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import numpy as np


# The goal of this project is to discover which players are the most
# valuable based on a few specific metrics and what age demographic they fall
# under. It also attempts to predict player value based on VoRP, BPM, Game 
#Score and more.Linear regression and simple data analysis are used in this
# program to help derive an answer.

# I referred to basketball reference for all statistical information in this 
# project. 

nba_stats = pd.read_csv(
    "25_normal_stats.csv")

advanced = pd.read_csv(
    "advanced_files.csv", usecols=[1,3,8,9,10,11,12,13,14,
                            15,16,17,18,19,20,21,22,23,24,25,
                            26,27])


filtered_stats = nba_stats[~nba_stats["Team"].str.contains("TM", na=False)]



merged = pd.merge(
    nba_stats,
    advanced,
    on=["Player", "Team"],
    how="inner",
)

# Dropping rows that split the player's season between two teams; I just want
# one total season from that player.

merged = merged.drop(index=[4,5,17,18,32,33,77,78,83,84,89,90,110,111,116,117,
                            138,139,140,149,150,163,164,170,171,189,190,199
                            ,200,205,206,221,222,224,225,233,234,246,247,258,
                            259,261,262,268,269,277,278,280,281,295,296,307,
                            308,310,311,315,316,319,320,330,331,338,339,345,
                            348,349,355,356,362,363,369,370,379,380,384,385,398
                            ,399,409,410,421,422,425,425,428,429,436,437,438])

merged = merged.astype({"Player":"string", "Team":"string", "Pos":"string",
                        "Awards":'string'})

# Making a second dataframe for players 
merged_2 = merged[merged["mp"] > 20]

# Filtering inconsistencies in the data. 

def clean_col(col):
    col = col.strip()
    col = col.replace(" ", "_")
    col = col.replace("(", "")
    col = col.replace(")", "")
    col = col.lower()
    return col

new_columns = []

for c in merged.columns:
    clean_c = clean_col(c)
    new_columns.append(clean_c)
    
merged.columns = new_columns


# Adding in Game Score and Fantasy Points for future analysis.

merged["avg_game_score"] = merged['pts'] + 0.4 * merged['fg'] - 0.7 * merged[
    'fga'] - 0.4*(merged['fta'] -merged['ft']) + 0.7 *merged[
        'orb'] + 0.3 * merged['drb'] + merged['stl'] + 0.7 * merged[
            'ast'] + 0.7 * merged['blk'] - 0.4 * merged['pf'] - merged['tov']
            
merged["fntsy_pts"] = merged["pts"] + merged["drb"] * 1.2 + merged[
    "ast"] * 1.5 + merged["stl"] * 3 + merged["blk"] * 3 - merged["tov"]

# Making a second dataframe for players who play 20+ minutes.

merged_2 = merged[merged["mp"] > 20]
                     

#%% Linear Regression model on Player Productivity and Player Worth


# The question that this attempts to answer may seem intuitive, but I
# was curious to see if increased productivity in terms of game score 
# would translate to higher value over a replacement value. 

x = merged["avg_game_score"]
y = merged["vorp"]

slope, intercept = np.polyfit(x, y, 1)
y_fit = slope * x + intercept

plt.scatter(x, y)
plt.plot(x, y_fit, color = 'black')
plt.xlabel("Average Game Score")
plt.ylabel("VoRP")
plt.title("Linear Regression: VoRP vs Avg Game Score")
plt.show()

X = sm.add_constant(x)
model = sm.OLS(y, X).fit()

model.summary()

# The result seems to show obvious positive correlation between the two 
# variables. I will now make this model more accurate by using the dataframe
# for players who average 20+ minutes a game.


x = merged_2["avg_game_score"]
y = merged_2["vorp"]

slope, intercept = np.polyfit(x, y, 1)
y_fit = slope * x + intercept

plt.scatter(x, y)
plt.plot(x, y_fit, color = 'black')
plt.xlabel("Average Game Score")
plt.ylabel("VoRP")
plt.title("Linear Regression: VoRP vs Avg Game Score")
plt.show()

X = sm.add_constant(x)
model = sm.OLS(y, X).fit()

model.summary()

# The correlation is even more accurate when specifying only players that play
# 20 minutes plus per game. Both models are skewed right, however, due to 
# a few players with ridiculously high productivity and value. I will use the
# latter dataframe for the rest of the analysis. 

##% What Influences Winning the Most?

# Average Game Score:

x = merged_2["avg_game_score"]
y = merged_2["ws"]

slope, intercept = np.polyfit(x, y, 1)
y_fit = slope * x + intercept

plt.scatter(x, y)
plt.plot(x, y_fit, color = 'black')
plt.xlabel("Average Game Score")
plt.ylabel("Win Shares")
plt.title("Linear Regression: Win Shares vs Avg Game Score")
plt.show()

X = sm.add_constant(x)
model = sm.OLS(y, X).fit()

model.summary()

# BPM:

x = merged_2["bpm"]
y = merged_2["ws"]

slope, intercept = np.polyfit(x, y, 1)
y_fit = slope * x + intercept


plt.scatter(x, y)
plt.plot(x, y_fit, color = 'red')
plt.xlabel("BPM")
plt.ylabel("Win Shares")
plt.title("Linear Regression: Win Shares vs BPM")
plt.show()

X = sm.add_constant(x)
model = sm.OLS(y, X).fit()

model.summary()

# VoRP

x = merged_2["vorp"]
y = merged_2["ws"]

slope, intercept = np.polyfit(x, y, 1)
y_fit = slope * x + intercept


plt.scatter(x, y)
plt.plot(x, y_fit, color = 'purple')
plt.xlabel("VoRP")
plt.ylabel("Win Shares")
plt.title("Linear Regression: Win Shares vs VoRP")
plt.show()

X = sm.add_constant(x)
model = sm.OLS(y, X).fit()

model.summary()

# Fantasy Points

x = merged_2["fntsy_pts"]
y = merged_2["ws"]

slope, intercept = np.polyfit(x, y, 1)
y_fit = slope * x + intercept


plt.scatter(x, y)
plt.plot(x, y_fit, color = 'purple')
plt.xlabel("Fantasy Points")
plt.ylabel("Win Shares")
plt.title("Linear Regression: Win Shares vs Fantasy Points")
plt.show()

X = sm.add_constant(x)
model = sm.OLS(y, X).fit()

model.summary()

# Conclusions From this Section: Based off linear regression, it seems clear 
# that VoRP is the most accurate predictor of Win Shares with an R-squared
# value of 79.8%. I used win shares as the indicator of player value because
# it has been shown to be an accurate measurer of player impact in the past.

# Fantasy points are the least reliable of these metrics to determine impact 
# on winning. There is certainly some correlation, but fantasy points should
# not be used to predict win shares. The reason I made these regression models
# was to show that those who rely on box score statistics and fantasy points
# to determine a player's value are faulty in their opinions. 

#%% Age Demographics

fix,ax = plt.subplots()
merged_2['age'].hist(bins=21, color='green', edgecolor='black')
plt.xlabel("Age of Players")
plt.ylabel("Count")
plt.show
plt.suptitle("Distribution of Player's Ages")

# We can gage from this graphic that the current NBA has the majority of its
# significant players in the 23-30 age range. This, of course, only includes 
# players that average 20 or more minutes per game. 

var =  merged_2.groupby("age")["pts"].mean()

fix,ax = plt.subplots()
ax.bar(var.index, var, color = 'Yellow')
ax.set_ylabel("Points Per Game")
ax.set_xlabel("Age")
plt.suptitle("PPG per Age")

# A few talented older players in the NBA make this graphic seem like NBA 
# players reach their scoring peak around ages 34-40, but realistically, their
# scoring peak is between 24-30. 

merged_2["total_points"] = (merged_2['pts'] * merged_2["g"]).round()

var2 =  merged_2.groupby("age")["total_points"].sum()

fix,ax = plt.subplots()
ax.bar(var2.index, var2, color = 'Yellow')
ax.set_ylabel("Points")
ax.set_xlabel("Age")
plt.suptitle("Points per Age")

# When it comes to total league scoring 22-29 year olds make up the vast 
# majority of total points.

# I will now filter for only player between the ages of 21-31 

merged_3 = merged_2[merged_2['age'] < 32]
merged_3 = merged_3[merged_3['age'] > 20]

var3 =  merged_3.groupby("age")["vorp"].mean()

fix,ax = plt.subplots()
ax.bar(var3.index, var3, color = 'Blue')
ax.set_ylabel("VoRP")
ax.set_xlabel("Age")
plt.suptitle("Avg. VoRP per Age")

var4 =  merged_3.groupby("age")["ws"].mean()

fix,ax = plt.subplots()
ax.bar(var4.index, var4, color = 'gray')
ax.set_ylabel("Win Shares")
ax.set_xlabel("Age")
plt.suptitle("Avg. Win Shares per Age")

# Although these graphics vary, we can determine that most of the NBA's 
# players that have the greatest impact on winning are from ages 26-30. 


