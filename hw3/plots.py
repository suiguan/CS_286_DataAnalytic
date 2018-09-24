# NAME: data_cleaning.py
"""
This file contains code for the programing assignment 
of module 2 week 2 
"""

# Imports
import sys, os
import numpy as np
import scipy as sp
import pandas as pd
from sklearn.preprocessing import Imputer
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

data_file = "m1_w2_ds1.csv"
def boxPlots(df, cols):
   nc = 3 #number of boxplots each row
   nr = len(cols) / nc #total number of columns
   if len(cols) % nc != 0: nr += 1

   fig = plt.figure()
   idx = 1
   for c in cols:
      ax = fig.add_subplot(nr, nc, idx)
      df.boxplot(column=[c,])
      ax.set_title(c, fontsize=8)
      ax.xaxis.label.set_size(8)
      ax.yaxis.label.set_size(8)
      idx += 1


def main():
   #1. Read in the data set
   df = pd.read_csv(data_file)

   #2. Replacing all occurrences of the string '1 5 255' with 0 
   df.replace('1 5 255', 0, inplace = True) 

   #3. convert ordinal values with numeric values 
   #"ordinalMap" is a dictionary, where 
   #key is the column name to be replaced, value is the categorical value to be mapped to 
   ordinalMap = {
      'PUBCHEM_TOTAL_CHARGE' : {'NEGATIVE':-1, 'ZERO':0, 'POSITIVE':1},
   } 
   for k in ordinalMap.keys(): df[k] = df[k].map(ordinalMap[k])

   #4. convert non-ordered categorical value to one-hot numeric
   #use get_dummies() to get one-hot, then use concat() to combine 
   df = pd.concat([df, pd.get_dummies(df['apol'])], axis=1)
   #drop the original categorical column
   df.drop('apol', axis=1, inplace=True)

   #5. imput missing value with mean of columns
   imr = Imputer(missing_values='NaN', strategy='mean', axis=0)
   imr = imr.fit(df)
   imputed_data = imr.transform(df.values)

   #6. saved the new imputed data back to the dataFrame
   for row in range(imputed_data.shape[0]):
      for column in range(imputed_data.shape[1]):
         df.iat[row, column] = imputed_data[row][column]

   #seaborn graph settings
   sns.set(style='whitegrid',  context='notebook')

   #7. scatter plot for first ten (10) columns and rows of the data set 
   cols = df.columns[:10]
   p1 = sns.pairplot(df[cols])
   p1.savefig("pairplot.png")

   #8. Heatmap for first ten (10) columns and rows of the data set 
   fig = plt.figure()
   cols = df.columns[:10]
   cm = np.corrcoef(df[cols].values.T)
   ax2 = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 5}, yticklabels=cols, xticklabels=cols)
   ax2.get_figure().savefig("heatmap.png")

   #9. Use boxplot to identify outliers
   print("plotting boxplot for 0 to 9 columns to identify outliers")
   boxPlots(df, df.columns[:9])
   print("plotting boxplot for 10 to 18 columns to identify outliers")
   boxPlots(df, df.columns[9:18])
   print("plotting boxplot for 19 and remaining columns to identify outliers")
   boxPlots(df, df.columns[18:])

   #10. use DataFrame.quantile() member function to get the quantiles range
   qdf = df.quantile([0, 0.25, 0.5, 0.75, 1.0])
   for c in qdf.columns:
      qs = qdf[c].values
      _min = qs[0]
      q1 = qs[1]
      q2 = qs[2]
      q3 = qs[3]
      _max = qs[4]
      #use IQR to determine if there is any outliers
      IQR = q3 - q1
      begin = q1 - (1.5 * IQR)
      end = q3 + (1.5 * IQR)
      if _min < begin or _max > end:
         print("column %s probably has outliers !! check them more closely!!" % c)
   
   #show image
   plt.tight_layout()
   plt.show()

if __name__ == '__main__':
   main()


