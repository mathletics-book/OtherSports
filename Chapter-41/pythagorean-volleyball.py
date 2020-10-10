###########################################################################
### Chapter 41- Volleyball Analytics                                    ###
# Mathletics: How Gamblers, Managers, and Sports Enthusiasts              #
# Use Mathematics in Baseball, Basketball, and Football                   #
###########################################################################

import pandas as pd
import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.model_selection import train_test_split
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

df = pd.read_csv("greekLeague1318-table.csv",delimiter="\t")

# we will define two objective functions
# one considering the sets won/lost
# one considering the total points
# this is the sum of the absolute differences between the predicted win and actual win %

def mad_sets(x,df):
	df['actual-wl']=df['Wins']/(df['Loses']+df['Wins'])
	df['ratio'] = df['Coef']
	return np.mean(abs(df['actual-wl']-(df['ratio']**x/(1+(df['ratio']**x)))))

def mad_points(x,df):
	df['actual-wl']=df['Wins']/(df['Loses']+df['Wins'])
	df['ratio'] = df['CoefPoints']
	return np.mean(abs(df['actual-wl']-(df['ratio']**x/(1+(df['ratio']**x)))))

## use a test set for out-of-sample predictions

train, test = train_test_split(df, test_size=0.3, random_state = 10)

res_sets = minimize_scalar(mad_sets, args = (train), bounds = (0,30), method = 'bounded')
res_points = minimize_scalar(mad_points, args = (train), bounds = (0, 30), method = 'bounded')

print("In sample (Sets): ", res_sets)
print("In sample (Points): ", res_points)

## make predictions for the test sets using the optimal coefficients

test['actual-wl']=test['Wins']/(test['Loses']+test['Wins'])
y_test_sets = test['Coef']**res_sets.x/(1+(test['Coef']**res_sets.x))
y_test_points = test['CoefPoints']**res_points.x/(1+(test['CoefPoints']**res_points.x))


print("Mean Absolute Out-of-Sample Error (Sets): ", np.mean(abs(test['actual-wl']-y_test_sets)))
print("Mean Absolute Out-of-Sample Error (Points): ", np.mean(abs(test['actual-wl']-y_test_points)))

fig = plt.figure()
ax1 = fig.add_subplot(111)


## add the y=x line
x = np.linspace(0,100,10000)
y = x
ax1.plot(x,y,'-.g',label="Perfect Prediction")

ax1.scatter(100*y_test_sets, 100*test['actual-wl'],s=20, c='b', marker="s", label='Sets')
ax1.scatter(100*y_test_points, 100*test['actual-wl'],s=20, c='r', marker="o", label='Points')


plt.title("Out-of-Sample Predictions")
plt.xlabel("Pythagorean Win %")
plt.ylabel("Actual Win %")
plt.legend(loc='upper left');
plt.show()

df = pd.read_csv("greekLeague1819-results.csv")

projected_wpcg = dict()

for r in range(np.max(df.Rd)):
	dftmp = df[df.Rd <= r+1] 
	setsfor = dict(zip(df.Home.unique(),np.zeros(len(df.Home.unique()))))
	setsagainst = dict(zip(df.Home.unique(),np.zeros(len(df.Home.unique()))))
	for g in range(len(dftmp)):
		setsfor[dftmp['Home'][g].strip()] += dftmp['HS'][g]
		setsagainst[dftmp['Home'][g].strip()] += dftmp['VS'][g]
		setsfor[dftmp['Visiting'][g].strip()] += dftmp['VS'][g]
		setsagainst[dftmp['Visiting'][g].strip()] += dftmp['HS'][g]
	for k in setsfor:
		if k not in projected_wpcg:
			projected_wpcg[k] = []
		projected_wpcg[k].append((setsfor[k]**res_sets.x)/((setsfor[k]**res_sets.x)+(setsagainst[k]**res_sets.x)))
	
actual_wp = dict(zip(df.Home.unique(),np.zeros(len(df.Home.unique()))))

for g in range(len(df)):
	if df['HS'][g] > df['VS'][g]:
		actual_wp[df['Home'][g].strip()] += 1/22.0
	else:
		actual_wp[df['Visiting'][g].strip()] += 1/22.0

error = lambda k: np.mean([abs(actual_wp[x]-projected_wpcg[x][k-1]) for x in actual_wp.keys()])

weekly_error = []

for w in list(range(5,23)):
	weekly_error.append(error(w))

fig = plt.figure()
ax1 = fig.add_subplot(111)
x = list(range(5,23))
ax1.plot(x,weekly_error)
plt.xticks(list(range(5,23)))

plt.xlabel("League Round")
plt.ylabel("Pythagorean Prediction Error")
plt.show()
