import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
 
df = pd.read_csv('FEV_Data.csv')
print(df.groupby('Location').describe())

print('Anova: Between(group, treatment), Within(residue, error)')
mod = ols('FEV ~ Location', data=df).fit()
aov_table = sm.stats.anova_lm(mod, typ=2)
print(aov_table)
