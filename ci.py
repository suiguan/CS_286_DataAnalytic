from scipy import stats
import numpy as np
import math

#data = np.array([107,119,99,116,101,121,114,120,104,88,114,124,152,100,125,114,95,117])

useT = True #True #set to True if population std is unknown
n = 11 #12 #data.shape[0]
xBar = 49.7485 #np.mean(data) 
std = 7.526921 #np.std(data)
CITwoSideConfident = 0.95

df = n - 1
print("%s 2-side CI with n=%s, xBar=%s, std=%s, df=%s" % (CITwoSideConfident, n, xBar, std, df))
probRight = (1 - CITwoSideConfident) / 2
if useT: 
   print("se = %s" % (std / math.sqrt(n),))
   t = stats.t.ppf(1-probRight, df)
   CI_0 = (-t * std / math.sqrt(n)) + xBar
   CI_1 = (t * std / math.sqrt(n)) + xBar
else: 
   z = stats.norm.ppf(1-probRight)
   CI_0 = (-z * std / math.sqrt(n)) + xBar
   CI_1 = (z * std / math.sqrt(n)) + xBar
print("%s: %s%% CI = (%s , %s)" % ("t-dis" if useT else "z-dis", CITwoSideConfident*100, CI_0, CI_1))

