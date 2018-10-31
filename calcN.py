from scipy import stats
import math

useT = False

alpha = 0.05 #Two-sided
beta = 0.1
ua = 3500
ub = 3800 #high side
std = 430 

za = stats.norm.ppf(1-(alpha/2))
zb = stats.norm.ppf(1-beta)
n = ((za+zb)*std/(ub-ua))**2
print("useT %d, n = %s" % (useT,n))
