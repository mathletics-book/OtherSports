import statsmodels.formula.api as smf
import pandas as pd 
import numpy as np

mls = pd.read_csv("MLSshots.csv")
f = 'Outcome~x + y + C(Assist) + C(Type) + C(Play) + Angle + Distance'

xG_mod = smf.glm(formula = f, data = mls, family = sm.families.Binomial()).fit()
print(logitfit.summary())
