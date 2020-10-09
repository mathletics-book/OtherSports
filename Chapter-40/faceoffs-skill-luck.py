import sys
import pandas as pd
import numpy as np
import math
import random

df = pd.read_csv("skaters1617.csv")

observed_var = np.var(df["FO%"])

t = []

for i in range(len(df["FOL"])):
	if (df["FOL"][i]+df["FOW"][i] > float(sys.argv[1]) and not math.isnan(df["FO%"][i])):
		t.append(df["FO%"][i])

observed_var = np.var(t)
# resample for the component of luck

resampled_fo = []

for i in range(len(df["FO%"])):
	if (df["FOL"][i]+df["FOW"][i] > float(sys.argv[1]) and not math.isnan(df["FO%"][i])):
		k = 0
		for j in range(df["FOW"][i]+df["FOL"][i]):
			if (random.random() < 0.5):
				k += 1.0
		resampled_fo.append(100*k/(df["FOW"][i]+df["FOL"][i]))

luck_var = np.var(resampled_fo)

print(observed_var,luck_var)	
