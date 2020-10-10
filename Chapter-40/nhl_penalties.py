import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from statsmodels.stats.proportion import proportions_ztest

penalties = pd.read_csv("penalties.csv").dropna()
carryover = np.zeros(len(penalties))
carryover[(penalties['starttime']> (18*60+10)) & (penalties['starttime'] < (20*60))] = 1
carryover[(penalties['starttime']> (38*60+10)) & (penalties['starttime'] < (40*60))] = 1
penalties['carryover'] = carryover
sns.distplot(penalties['starttime'], hist = False, kde = True, kde_kws = {'linewidth': 3})
plt.show()

## proportion test for carryover - vs - non carryover 

print("t,carryover_rate, non_carryover_rate, p-val")

for t in [10, 30, 60, 90]:

    ncarryobs = len(penalties[(penalties.carryover == 0) ])
    ncarrysuc = len((penalties[(penalties.carryover == 0) & (penalties.goal == 1) ]))
    carryobs = len(penalties[(penalties.carryover == 1) & ((penalties.starttime > 18*60+t) | (penalties.starttime > 38*60+t))])
    carrysuc = len((penalties[(penalties.carryover == 1) & (penalties.goal == 1) & ((penalties.starttime > 18*60+t) | (penalties.starttime > 38*60+t))]))

    stat, pval = proportions_ztest([ncarrysuc,carrysuc], [ncarryobs,carryobs])
    print(120-t, carrysuc/carryobs, ncarrysuc/ncarryobs, pval)
    
# power play degree/skaters

penalties['skater_diff'] = abs(penalties['nskatersh']-penalties['nskatersa'])
print(np.mean(penalties[penalties.carryover == 0]['skater_diff']))
print(np.mean(penalties[penalties.carryover == 1]['skater_diff']))
ttest_ind(np.array(penalties[penalties.carryover == 0]['skater_diff']),np.array(penalties[penalties.carryover == 1]['skater_diff']))

## shorthanded goals

print("t,carryover_sgoal_rate, non_carryover_sgoal_rate, p-val")

for t in [10, 30, 60, 90]:

    ncarryobs = len(penalties[(penalties.carryover == 0) ])
    ncarrysuc = len((penalties[(penalties.carryover == 0) & (penalties.goal == -1) ]))
    carryobs = len(penalties[(penalties.carryover == 1) & ((penalties.starttime > 18*60+t) | (penalties.starttime > 38*60+t))])
    carrysuc = len((penalties[(penalties.carryover == 1) & (penalties.goal == -1) & ((penalties.starttime > 18*60+t) | (penalties.starttime > 38*60+t))]))

    stat, pval = proportions_ztest([ncarrysuc,carrysuc], [ncarryobs,carryobs])
    print(120-t, carrysuc/carryobs, ncarrysuc/ncarryobs, pval)
