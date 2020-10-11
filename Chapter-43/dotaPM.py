import pickle
import json
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.metrics import accuracy_score,brier_score_loss
from sklearn.calibration import calibration_curve

## adopted from the RAPM code from Ryan Davis: https://github.com/rd11490/NBA_Tutorials

def build_hero_list(posessions):
	heros = list(set(list(posessions['H1Id'].unique()) + list(posessions['H2Id'].unique()) + list(posessions['H3Id']) + list(posessions['H4Id'].unique()) + list(posessions['H5Id'].unique()) + list(posessions['H6Id'].unique()) +list(posessions['H7Id'].unique()) + list(posessions['H8Id'].unique()) + list(posessions['H9Id'].unique()) + list(posessions['H10Id'].unique())))
	heros.sort()
	return heros

def heros2vector(row_in, heros):
	p1, p2, p3, p4, p5 = row_in[0:5]
	p6, p7, p8, p9, p10 = row_in[5:10]
	rowOut = np.zeros([len(heros)])
	rowOut[[heros.index(p) for p in [p1, p2, p3, p4, p5]]] += 1
	rowOut[[heros.index(p) for p in [p6, p7, p8, p9, p10]]] += -1
	return rowOut

def datadf2mat(games, yvar, heros):
	stints_x_base = games.as_matrix(columns=['H1Id', 'H2Id','H3Id', 'H4Id', 'H5Id','H6Id', 'H7Id', 'H8Id','H9Id', 'H10Id'])
	stint_X_rows = np.apply_along_axis(heros2vector, 1, stints_x_base, heros)
	stint_Y_rows = games.as_matrix([yvar])
	return stint_X_rows, stint_Y_rows 

def lambda_to_alpha(lambda_value, samples):
	return (lambda_value * samples) / 2.0

def alpha_to_lambda(alpha_value, samples):
	return (alpha_value * 2.0) / samples

### this is the main function that runs the ridge regression
def calculate_rapm(train_x, train_y, lambdas, varname, heros):
	alphas = [lambda_to_alpha(l, train_x.shape[0]) for l in lambdas]
	clf = RidgeCV(alphas=alphas, cv=5, fit_intercept=True, normalize=False)
	model = clf.fit(train_x, train_y) 
	hero_arr = np.transpose(np.array(heros).reshape(1, len(heros)))
	coef_offensive_array = np.round(np.transpose(model.coef_[:, 0:len(heros)]),3)
	hero_id_with_coef = np.concatenate([hero_arr, coef_offensive_array], axis=1)
	heros_coef = pd.DataFrame(hero_id_with_coef)
	intercept = model.intercept_
	heros_coef.columns = ['heroId', '{0}__PM'.format(varname)]
	heros_coef['{0}__intercept'.format(varname)] = intercept[0]
	return heros_coef, intercept


df_all = pd.read_csv("Data/features.csv")
df_target = pd.read_csv("Data/targets.csv")

df = df_all[['r1_hero_id','r2_hero_id','r3_hero_id','r4_hero_id','r5_hero_id','d1_hero_id','d2_hero_id','d3_hero_id','d4_hero_id','d5_hero_id']]
df['radiant_win'] = df_target['radiant_win']
df['rad_win'] = df.radiant_win.astype(int)
df.columns = ['H1Id','H2Id','H3Id','H4Id','H5Id','H6Id','H7Id','H8Id','H9Id','H10Id','radiant_win','rad_win'] 

## find replacement heros; defined as heros chosen at the bottom 20% quantile

freq = df[['H1Id','H2Id','H3Id','H4Id','H5Id','H6Id','H7Id','H8Id','H9Id','H10Id']].stack().value_counts().to_dict()
replacement = []
for k in freq:
	if freq[k] < np.quantile(list(freq.values()),0.2):
		replacement.append(k)
for i in range(len(df)):
	for c in ['H1Id','H2Id','H3Id','H4Id','H5Id','H6Id','H7Id','H8Id','H9Id','H10Id']:
		if df.iloc[i][c] in replacement:
			df.at[i,c] = 0

hero_list = build_hero_list(df)


df_train = df.sample(frac=0.8,random_state=200)
df_test = df.drop(df_train.index)
df = df_train

yvar = 'rad_win'
train_x, train_y = datadf2mat(df, yvar, hero_list)


lambdas_rapm = list(np.arange(0.1, 1, 0.1))
results, intercept = calculate_rapm(train_x, train_y, lambdas_rapm, yvar, hero_list)

#print(results)


apm = dict()

for i in range(len(results)):
        apm[int(results.iloc[i]['heroId'])] = results.iloc[i]['rad_win__PM']



for k in apm:
	print(k,apm[k])

## predictions 

df_test = df_test.reset_index()

pred = []

for i in range(len(df_test)):
        rad = apm[df_test.iloc[i]['H1Id']]+apm[df_test.iloc[i]['H2Id']]+apm[df_test.iloc[i]['H3Id']]+apm[df_test.iloc[i]['H4Id']]+apm[df_test.iloc[i]['H5Id']]
        dire = apm[df_test.iloc[i]['H6Id']]+apm[df_test.iloc[i]['H7Id']]+apm[df_test.iloc[i]['H8Id']]+apm[df_test.iloc[i]['H9Id']]+apm[df_test.iloc[i]['H10Id']]
        pred.append(rad-dire)

pred += intercept
print(intercept)
print(accuracy_score(np.array(df_test.rad_win),np.round(pred)))
print(brier_score_loss(df_test.rad_win,pred))
print(calibration_curve(df_test.rad_win, pred,n_bins=10,strategy="quantile"))

