# Imports
import sys, os
import numpy as np
import scipy as sp
import pandas as pd
from scipy import stats
from sklearn.preprocessing import Imputer

data_file = "unicef.csv"

def main():
   #a. Read in the data set
   df = pd.read_csv(data_file)

   #b. Replacing all occurrences of missing data with NaN
   df.replace('.', np.NaN, inplace = True) 

   #Compute the mean and the median for column "lowbwt"
   #drop rows if there are any NaN
   arr = pd.to_numeric(df.dropna(axis=0)['lowbwt']).values
   mean = np.mean(arr) 
   median = np.median(arr) 
   tmean = stats.trim_mean(arr, 0.05)
   print(mean, median, tmean)

   #drop rows if there are any NaN
   arr = pd.to_numeric(df.dropna(axis=0)['life60']).values
   mean = np.mean(arr) 
   median = np.median(arr) 
   tmean = stats.trim_mean(arr, 0.05)
   print(mean, median, tmean)

   #drop rows if there are any NaN
   arr = pd.to_numeric(df.dropna(axis=0)['life92']).values
   mean = np.mean(arr) 
   median = np.median(arr) 
   tmean = stats.trim_mean(arr, 0.05)
   print(mean, median, tmean)

if __name__ == '__main__':
   main()
