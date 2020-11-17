import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np

mls = pd.read_csv("MLSshots.csv")
f = 'Outcome~x + y + C(Assist) + C(Type) + C(Play) + Angle + Distance'

X_train, X_test, y_train, y_test = train_test_split(mls, mls['Outcome'], test_size=0.2)
xG_mod = smf.glm(formula = f, data = X_train, family = sm.families.Binomial()).fit()
print(xG_mod.summary())


## this computes the calibration curve out-of-sample

observed, predicted = calibration_curve(X_test['Outcome'], xG_mod.predict(X_test), n_bins = 10)

fig = plt.figure()
ax1 = fig.add_subplot(111)
x = np.linspace(0,1,10000)
y = x
ax1.plot(x,y,'-.g',label="Perfect Calibration")
ax1.scatter(predicted,observed,s=20, c='b', marker="s", label = "xG Model")
plt.xlabel("Predicted Shot Make Probability")
plt.ylabel("Observed Shot Make Probability")
plt.legend(loc='upper left')
plt.show()
