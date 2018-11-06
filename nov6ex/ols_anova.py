import pandas as pd

# For statistics. Requires statsmodels 5.0 or more
from statsmodels.formula.api import ols
# Analysis of Variance (ANOVA) on linear models
from statsmodels.stats.anova import anova_lm
 
df = pd.read_csv('Guan_data.csv')

# Fit the model
#model = ols("resp ~ x1 + x2 + x3 + x4", df).fit()
model = ols("resp ~ x1 + x3 + x4 -1", df).fit()

# Print the summary
print(model.summary())

# Peform analysis of variance on fitted linear model
anova_results = anova_lm(model)

print('\nANOVA results')
print(anova_results)
