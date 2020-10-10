###########################################################################
### Chapter 41 - Volleyball Analytics                                   ###
# Mathletics: How Gamblers, Managers, and Sports Enthusiasts              #
# Use Mathematics in Baseball, Basketball, and Football                   #
###########################################################################

import pandas as pd
import numpy as np
import warnings
import math
import random
from transliterate import translit
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

df = pd.read_csv("greekLeague1819-results.csv")
teams = list(np.sort(list(df.Home.unique())))
## we will use the first 15 rounds of the league to build the BT model and predict the rest 7 rounds.

df_train = df[df.Rd <= 15]
df_test = df[df.Rd > 15]
df_test = df_test.reset_index()

# find how many sets in total in our training set

n_sets = 0

for g in range(len(df_train)):
	n_sets += min(4,df_train.HS[g]+df_train.VS[g])

X = np.zeros((n_sets,len(df_train.Home.unique())))
y = np.zeros(n_sets)

# we will go over every set and mark the home team with a +1 and the visiting team with a -1
# the result will be 1 if the home team won the set and 0 if the home team lost the set

n_sets = 0

for g in range(len(df_train)):
	if df_train.HS[g]+df_train.VS[g] == 5:
		for s in range(2):
			X[n_sets,teams.index(df_train.Home[g].strip())] = 1
			X[n_sets,teams.index(df_train.Visiting[g].strip())] = -1
			y[n_sets] = 1
			n_sets += 1
		for s in range(2):
			X[n_sets,teams.index(df_train.Home[g].strip())] = 1
			X[n_sets,teams.index(df_train.Visiting[g].strip())] = -1
			n_sets += 1
	else:
		for s in range(df_train.HS[g]):
			X[n_sets,teams.index(df_train.Home[g].strip())] = 1
			X[n_sets,teams.index(df_train.Visiting[g].strip())] = -1
			y[n_sets] = 1
			n_sets += 1
		for s in range(df_train.VS[g]):
			X[n_sets,teams.index(df_train.Home[g].strip())] = 1
			X[n_sets,teams.index(df_train.Visiting[g].strip())] = -1
			n_sets += 1

model = LogisticRegression()
model.fit(X, y)

team_abilities = dict()

for t in teams:
	team_abilities[t] = model.coef_[0][teams.index(t)]

home_edge = model.intercept_[0]

print("=========== Team Ratings ===========")
print("Home edge: ", round(home_edge,4))
print("------------------------------------")
print("                Team   Rating")
for i, t in enumerate(teams):
	print("{:>20s}    {:.4f}".format(translit(t,'el',reversed = True), team_abilities[t]))
print("------------------------------------")


### make predictions for the last 7 rounds
### Simulate every game 1000 times using the ratings to simulate the sets of each game

home_wp = []
result = []
n_sim = 1000

for g in range(len(df_test)):
	result.append(1 if df_test.HS[g] > df_test.VS[g] else 0)
	home_wins_sim = 0
	home_setability = team_abilities[df_test.Home[g].strip()]
	away_setability = team_abilities[df_test.Visiting[g].strip()]
	home_setwin_response = home_edge + home_setability - away_setability
	home_setwp = math.exp(home_setwin_response)/(1+math.exp(home_setwin_response)) 
	for s in range(n_sim):
		hsets = 0
		asets = 0
		for sets in range(4):
			if (random.random() < home_setwp):
				hsets += 1
			else:
				asets += 1
			if hsets == 3:
				home_wins_sim += 1
				break
			if asets == 3:
				break
		if hsets == 2 and asets == 2:
			if (random.random() < 0.5):
				home_wins_sim+=1
	home_wp.append(home_wins_sim/float(n_sim))

accuracy = sum([int(round(home_wp[x]) == result[x]) for x in range(len(result)) ])/float(len(result))

print("Prediction accuracy: ", accuracy)

observed, predicted = calibration_curve(result, home_wp,n_bins = 5)

fig = plt.figure()
ax1 = fig.add_subplot(111)

x = np.linspace(0,1,10000)
y = x
ax1.plot(x,y,'-.g',label="Perfect Calibration")

ax1.scatter(predicted,observed,s=20, c='b', marker="s", label = "Team Ratings Calibration")


plt.xlabel("Predicted Win Probability")
plt.ylabel("Observed Win Probability")
plt.legend(loc='upper left');
plt.show()
