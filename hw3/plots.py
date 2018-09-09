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

data_file = "m1_w2_ds1.csv"
def getMeanStd(imputed_data):
   columnMean = np.mean(imputed_data, axis=0)
   rowMean = np.mean(imputed_data, axis=1)
   columnStd = np.std(imputed_data, axis=0)
   rowStd = np.std(imputed_data, axis=1)
   return columnMean, rowMean, columnStd, rowStd

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
   #cols = df.columns[:10]
   #p1 = sns.pairplot(df[cols])
   #p1.savefig("pairplot.png")

   #8. Heatmap for first ten (10) columns and rows of the data set 
   #cols = df.columns[:10]
   #cm = np.corrcoef(df[cols].values.T)
   #ax2 = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 5}, yticklabels=cols, xticklabels=cols)
   #ax2.get_figure().savefig("heatmap.png")
   
   #show image
   plt.show()

if __name__ == '__main__':
   main()


