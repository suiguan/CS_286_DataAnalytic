"""
NAME: suikun_guan_chk_data_normality.py 

This module is for Programming Assignment 3 to quantify the degree of difference
between the given data distribution and normal distribution. 

This module is based on the "cs286_chk_norm_dist.py" by Professor Leonard Wesley @FALL 2018 CS 286 SecII under Week10/code
"""

# Imports
import argparse
import numpy as np
import pandas as pd
import itertools
import pdb
import math
import sys
from decimal import *
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.stats.mstats import normaltest
from textwrap import wrap


__NUM_ITERATIONS__ = 1 
__MATPLOTLIP_TITLE_WIDTH__ = 60
__PLOT_SIZE_X__ = 8
__PLOT_SIZE_Y__ = 6
__TRAINING_TEST_SPLIT__ = 0.8

# helper functions:
# Compute vector sum
def comp_vec_sum(nums):
   return np.sqrt(np.sum([x*x for x in nums])) 


# the main module method to compare the distribution
# reference: cs286_chk_norm_dist.py by Professor Leonard Wesley @FALL 2018 CS 286 SecII under Week10/code
# most of the codes are same from the reference, except I do not included some of the debug prints, and
# read file is based on the arguments instead of the fixed filename "example_dataset.csv"
def compare(dataset):
   df = pd.read_csv(dataset)
   df_num_rows = len(df.index)
   df_num_cols = len(df.columns)

   # Calculate number of samples to use as an example training set 
   # for which the degree to which it is a normal distribution 
   # will be determined. Must have at least two samples
   num_samples = max(2, int(df_num_rows * __TRAINING_TEST_SPLIT__)) 

   print("Starting to compute the degree of match between ")
   print(" a training and test data sets over ", __NUM_ITERATIONS__, " iteration(s)")
   iter_ctr = 1
   fig_ctr = 1
   for _ in itertools.repeat(None, __NUM_ITERATIONS__): 
     # Randomly select num_samples from df
     new_df = df.sample(n=num_samples)
     
     new_df_num_rows = len(new_df.index)
     new_df_num_cols = len(new_df.columns)
     
     # Extract trainig and test data sets
     dfsvc_train = df.sample(frac = __TRAINING_TEST_SPLIT__)
     dfsvc_test = pd.concat([dfsvc_train, df]).loc[dfsvc_train.index.symmetric_difference(df.index)] 
     
     # Training data
     __PREDICTOR_VARIABLES__ = df.columns[2:]
     X = dfsvc_train[__PREDICTOR_VARIABLES__]
     X_test = dfsvc_test[__PREDICTOR_VARIABLES__] 
     
     # Scale the data set from -1 to 1
     print ("\n\n   Scaling training data set between [-1., 1.]" )
     scaler = MinMaxScaler(feature_range = (-1., 1.))
     X_scaled = scaler.fit_transform(X)
     X_test_scaled = scaler.fit_transform(X_test)


     # Generate histograms for both classes in both the training and test data sets
     # First compute vector sum of samples for training set
     print("   Deterining the degree of fit between training and test data to a normal distribution.")
     col_names = X.columns
     df_X_scaled = pd.DataFrame(X_scaled, columns = col_names)
     df_X_test_scaled = pd.DataFrame(X_test_scaled, columns = col_names)
     
     # Make copy of data frames and compute vector sum in preparation to 
     # generate histograms
     df_X_scaled_vecsum = df_X_scaled
     df_X_test_scaled_vecsum = df_X_test_scaled
     df_X_scaled_vecsum['vec_sum'] = df_X_scaled_vecsum.apply(comp_vec_sum, axis = 1)
     df_X_test_scaled_vecsum['vec_sum'] = df_X_test_scaled_vecsum.apply(comp_vec_sum, axis = 1)
     
     # Determine fit of training and test data to a normal distribution
     # That is, test the underlying assumption of the VC Dimension that 
     # a normal disgtribution governs the distribution of the data. 
     """
     scipy.stats.mstats.normaltest:
         Tests whether a sample differs from a normal distribution. 
         This function tests the null hypothesis that a sample comes from a normal distribution. 
         It is based on D’Agostino and Pearson’s [R246], [R247] test that combines skew and 
         kurtosis to produce an omnibus test of normality. 
         
         Parameters:	
             a : array_like
                 The array containing the data to be tested.
             axis : int or None
                 If None, the array is treated as a single data set, regardless 
                 of its shape. Otherwise, each 1-d array along axis axis is tested.
             Returns:k2 : float or array Values range from 0 to 100, where 100 = perfect match 
                         to normal dist
                 s^2 + k^2, where s is the z-score returned by skewtest and k 
                 is the z-score returned by kurtosistest, where kurtosis = 3(n-1)/(n+1).
                 
                     p-value : float or array
                         A 2-sided chi squared probability for the hypothesis test. 
     
         REFS:                 
         [R246] D’Agostino, R. B. (1971), “An omnibus test of normality for moderate 
             and large sample size,” Biometrika, 58, 341-348
         [R247] D’Agostino, R. and Pearson, E. S. (1973), “Testing for departures from 
             normality,” Biometrika, 60, 613-622
     """
     # Extract the vector sum info from the train and test data sets
     X_scaled_hist_data = df_X_scaled_vecsum['vec_sum']
     X_test_scaled_hist_data = df_X_test_scaled_vecsum['vec_sum']
     
     # Compute degree of match of data to normal dist
     X_scaled_hr = normaltest(X_scaled_hist_data)
     X_scaled_hr_match = X_scaled_hr[0]
     X_scaled_hr_match_pvalue = X_scaled_hr[1]
     X_test_scaled_hr = normaltest(X_test_scaled_hist_data)
     X_test_scaled_hr_match = X_test_scaled_hr[0]
     X_test_scaled_hr_match_pvalue = X_test_scaled_hr[1]
     print("   Training data set match to normal dist: %.1f  with p-value: %.4E" % \
             (X_scaled_hr_match, Decimal(X_scaled_hr_match_pvalue)))
     print("   Test data set match to normal dist:     %.1f  with p-value: %.4E" % \
             (X_test_scaled_hr_match, Decimal(X_test_scaled_hr_match_pvalue)))
     print("Completed deterining the degree of fit of training and test data to normal distribution")
     print("  for iteration: ", iter_ctr)
     
     # Display histograms for training and test data
     # See:  http://danielhnyk.cz/fitting-distribution-histogram-using-python/ 
     print("Displaying histograms for training and test data sets.")
     
     # Display training data first
     fig = plt.figure(fig_ctr, figsize = (__PLOT_SIZE_X__, __PLOT_SIZE_Y__)) 
     fig_ctr = 1 + fig_ctr
     plt.gcf().clear()
             
     X_scaled_hist_data.hist(normed = True)
     X_scaled_hist_data.plot(kind = 'kde', linewidth = 2, \
                             color = 'r', label = 'Distribution Of Training Data')
     
     # find minimum and maximum of xticks, so we know
     # where we should compute theoretical distribution
     xt = plt.xticks()[0]  
     xmin, xmax = min(xt), max(xt)  
     lnspc = np.linspace(xmin, xmax, len(X_scaled_hist_data))
     
     # Now display the normal distribution over the histogram of the 
     # training data
     m, s = stats.norm.fit(X_scaled_hist_data) # get mean and standard deviation  
     pdf_g = stats.norm.pdf(lnspc, m, s) # now get theoretical values in our interval  
     plt.plot(lnspc, pdf_g, label="Normal Distribution", color = 'k', linewidth = 2) # plot it
     
     plt.xlabel("Training data feature vector distance/magnitude.")
     plt.ylabel("Frequency.")
     match_val = '%.2f' % Decimal(X_scaled_hr_match)
     match_p_val = '%.4E' % Decimal(X_scaled_hr_match_pvalue)
     
     title_str = "Histrogram and Distribution of training data overlayed with normal distribution. " \
        + "  Degree of match = " + match_val + " with p-value = " + match_p_val + "."
     plt.title("\n".join(wrap(title_str, __MATPLOTLIP_TITLE_WIDTH__)))
     
     leg = plt.legend(loc = 'best', ncol = 1, shadow = True, fancybox = True)
     leg.get_frame().set_alpha(0.5)
     
     plt.show()
         
     # Display test dataset next
     fig = plt.figure(fig_ctr, figsize = (__PLOT_SIZE_X__, __PLOT_SIZE_Y__))
     fig_ctr = 1 + fig_ctr 
     plt.gcf().clear()
             
     X_test_scaled_hist_data.hist(normed = True)
     X_test_scaled_hist_data.plot(kind = 'kde', linewidth = 2, \
                             color = 'r', label = 'Distribution Of Test Data')
     
     # find minimum and maximum of xticks, so we know
     # where we should compute theoretical distribution
     xt = plt.xticks()[0]  
     xmin, xmax = min(xt), max(xt)  
     lnspc = np.linspace(xmin, xmax, len(X_test_scaled_hist_data))
     
     # Now display the normal distribution over the histogram of the test data
     m, s = stats.norm.fit(X_test_scaled_hist_data) # get mean and standard deviation  
     pdf_g = stats.norm.pdf(lnspc, m, s) # now get theoretical values in our interval  
     plt.plot(lnspc, pdf_g, label="Normal Distribution", color = 'k', linewidth = 2) # plot it
     
     plt.xlabel("Test data feature vector distance/magnitude.")
     plt.ylabel("Frequency.")
     match_val = '%.2f' % Decimal(X_test_scaled_hr_match)
     match_p_val = '%.4E' % Decimal(X_test_scaled_hr_match_pvalue) 
     
     title_str = "Histogram and Distribution of test data overlayed with normal distribution." \
        + "  Degree of match = " + match_val + " with p-value = " + match_p_val + "."
     
     plt.title("\n".join(wrap(title_str, __MATPLOTLIP_TITLE_WIDTH__)))
     
     leg = plt.legend(loc = 'best', ncol = 1, shadow = True, fancybox = True)
     leg.get_frame().set_alpha(0.5)
     
     plt.show() 
     print("Completed displaying histograms for training and test data sets")
     print("  for iteration: ", iter_ctr)
     
     # Increment iteration count
     iter_ctr = 1 + iter_ctr 
     
     if iter_ctr <= __NUM_ITERATIONS__:
        print("")
        print("Starting iteration: ", iter_ctr) 
     else:
        print()        
        print(">>>>> DONE <<<<<")




def main(): 
   parser = argparse.ArgumentParser()
   parser.add_argument('-n', help='name of the data set', required=True, type=str)
   args = parser.parse_args()
   compare(args.n)

if __name__ == '__main__': 
    main() 
