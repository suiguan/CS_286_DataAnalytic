# NAME: data_cleaning.py
"""
This file contains code for the programing assignment 
of week 2 data cleaning
"""

# Imports
import sys, os
import numpy as np
import scipy as sp
import pandas as pd
from sklearn.preprocessing import Imputer

data_file = "m1_w2_ds1.csv"
def getMeanStd(imputed_data):
   columnMean = np.mean(imputed_data, axis=0)
   rowMean = np.mean(imputed_data, axis=1)
   columnStd = np.std(imputed_data, axis=0)
   rowStd = np.std(imputed_data, axis=1)
   return columnMean, rowMean, columnStd, rowStd

def main():
   #a. Read in the data set
   print("Step A: reading file %s" % data_file) 
   df = pd.read_csv(data_file)
   print("Step A: finished reading file %s\n" % data_file) 

   #b. Replacing all occurrences of the string '1 5 255' with 0 
   print ("Step B: Replacing '1 5 255' in data set with 0", df)
   df.replace('1 5 255', 0, inplace = True) 
   print ("Step B: Completed replacing '1 5 255'\n", df)

   #c. convert ordinal values with numeric values 
   #"ordinalMap" is a dictionary, where 
   #key is the column name to be replaced, value is the categorical value to be mapped to 
   print("Step C: converting ordinal values")
   ordinalMap = {
      'PUBCHEM_TOTAL_CHARGE' : {'NEGATIVE':-1, 'ZERO':0, 'POSITIVE':1},
   } 
   #print("df BEFORE:\n", df)
   for k in ordinalMap.keys(): df[k] = df[k].map(ordinalMap[k])
   #print("df AFTER:\n", df)
   print("Step C: Completed converting ordinal values.\n")

   #d. convert non-ordered categorical value to one-hot numeric
   print("Step D: converting categorical values to one-hot")
   #print("df BEFORE:\n", df)
   #use get_dummies() to get one-hot, then use concat() to combine 
   df = pd.concat([df, pd.get_dummies(df['apol'])], axis=1)
   #drop the original categorical column
   df.drop('apol', axis=1, inplace=True)
   #print("df AFTER:\n", df)
   print("Step D: Completed converting categorical values.\n")

   #e1. imput missing value with mean of columns
   print("Step E1: Imputing missing data with mean of columns")
   imr = Imputer(missing_values='NaN', strategy='mean', axis=0)
   imr = imr.fit(df)
   imputed_data = imr.transform(df.values)
   #print("imputed df:\n", imputed_data)
   print("Step E1: Completed imputing missing data\n")

   #f1. print mean, std for each column and row
   print("Step F1: printing column and row mean & std")
   columnMean, rowMean, columnStd, rowStd = getMeanStd(imputed_data) 
   for c in range(columnMean.shape[0]): print("column %d: mean = %s, std = %s" % (c, columnMean[c], columnStd[c]))
   print()
   for r in range(rowMean.shape[0]): print("row %d: mean = %s, std = %s" % (r, rowMean[r], rowStd[r]))
   print()

   #e2. eliminate all rows if there is a missing value
   print("Step E2: instead of imputing missing data, drop rows that have any missing values (NaN)")
   #drop columns where all data is NaN first
   df = df.dropna(axis=1, how='all')
   #drop rows if there are any NaN
   df = df.dropna(axis=0)
   print("Step E2: complete drop rows with NaN")

   #f2. print mean, std for each column and row
   print("Step F2: printing column and row mean & std")
   columnMean, rowMean, columnStd, rowStd = getMeanStd(df.values) 
   for c in range(columnMean.shape[0]): print("column %d: mean = %s, std = %s" % (c, columnMean[c], columnStd[c]))
   print()
   for r in range(rowMean.shape[0]): print("row %d: mean = %s, std = %s" % (r, rowMean[r], rowStd[r]))
   print()


if __name__ == '__main__':
   main()


