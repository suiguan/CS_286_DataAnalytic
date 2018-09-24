# Imports
import sys, os
import numpy as np
import scipy as sp
import pandas as pd
from scipy import stats
import math


df = pd.read_csv('serzinc.csv')
info = df.describe()
print(info)

count = info.values[0][0]
mean = info.values[1][0]
std = info.values[2][0]
print("count %s, mean %s, std %s" % (count, mean, std))

ci = 0.90
df = count - 1
inter = stats.t.interval(ci, df)
print("%s CI (t, df = %s) = %s" % (ci, df, inter))

sampleMean = mean
sampleStd = std


#t = (x - sampleMean) / (sampleStd / math.sqrt(count)) 
start = (inter[0] * sampleStd / math.sqrt(count)) + sampleMean
end = (inter[1] * sampleStd / math.sqrt(count)) + sampleMean
print("%s CI (with sample mean %s, sample std %s) = %s" % (ci, sampleMean, sampleStd, (start, end)))
