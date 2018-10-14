# NAME: cal_n.py
"""
This file contains code for the programing assignment 
of week 8 to calculate sample size

This python module uses the following package:
- scipy: to lookup z-value from a given probability.
- argparse: to parse command line arguments.
"""

# Imports
from scipy import stats
import argparse
import sys

def cal_n(alpha, beta, m, mn, std, isOneTail): 
   prob = alpha
   if not isOneTail: #2-tail test
      prob = alpha/2.0
   za = stats.norm.ppf(1-prob)
   zb = stats.norm.ppf(1-beta)
   n = ((za+zb)*std/(mn-m))**2
   print(n)
   return n

if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument('-a', help='alpha used in hypothesis testing', required=True, type=float)
   parser.add_argument('-b', help='beta, the probability of making type II error', required=True, type=float)
   parser.add_argument('-m', help='inferred population mean from the sample mean', required=True, type=float)
   parser.add_argument('-mn', help='population mean can be as high (or as low) as this value', required=True, type=float)
   parser.add_argument('-std', help='standard deviation', required=True, type=float)
   parser.add_argument('-f', help='T/F: whether to use 1-tail (T) or 2-tail test (F)', required=True)
   args = parser.parse_args()
   if args.f != 'T' and args.f != 'F':
      parser.print_help() 
      sys.exit(1)

   cal_n(args.a, args.b, args.m, args.mn, args.std, args.f == "T") 
