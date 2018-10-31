import sys, os
import numpy as np
import scipy as sp
import pandas as pd
from sklearn.preprocessing import Imputer
import statsmodels.formula.api as sm
from statsmodels import tools
from sklearn.linear_model import LinearRegression
import scipy, scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn') # pretty matplotlib plots


data_file = 'lowbwt.csv'

if __name__ == "__main__":
   df = pd.read_csv(data_file)
   print(df[['sbp', 'gestage']].groupby('gestage').describe())
   #bx = df.plot(kind='scatter', x='gestage',y='sbp')
   #plt.show()

   #OLS
   Y = df['sbp']
   X = df[['gestage']]
   X = tools.add_constant(X)
   result = sm.OLS(Y, X).fit()

   ols = result.summary()
   print(ols)

   print("params: " , result.params)

   # fitted values (need a constant term for intercept)
   model_fitted_y = result.fittedvalues

   # model residuals
   model_residuals = result.resid

   # normalized residuals
   model_norm_residuals = result.get_influence().resid_studentized_internal

   # absolute squared normalized residuals
   model_norm_residuals_abs_sqrt = np.sqrt(np.abs(model_norm_residuals))

   # absolute residuals
   model_abs_resid = np.abs(model_residuals)

   # leverage, from statsmodels internals
   model_leverage = result.get_influence().hat_matrix_diag

   # cook's distance, from statsmodels internals
   model_cooks = result.get_influence().cooks_distance[0]

   plot_lm_1 = plt.figure(1)
   plot_lm_1.set_figheight(8)
   plot_lm_1.set_figwidth(12)

   plot_lm_1.axes[0] = sns.residplot(model_fitted_y, 'sbp', data=df, 
                             lowess=True, 
                             scatter_kws={'alpha': 0.5}, 
                             line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})

   plot_lm_1.axes[0].set_title('Residuals vs Fitted')
   plot_lm_1.axes[0].set_xlabel('Fitted values')
   plot_lm_1.axes[0].set_ylabel('Residuals')

   # annotations
   abs_resid = model_abs_resid.sort_values(ascending=False)
   abs_resid_top_3 = abs_resid[:3]

   for i in abs_resid_top_3.index:
       plot_lm_1.axes[0].annotate(i, xy=(model_fitted_y[i], model_residuals[i]));

   plt.show()
