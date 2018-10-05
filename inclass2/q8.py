from scipy import stats
import numpy as np
import math

x = 195
u = 183
std = 37 / math.sqrt(60)
z = (x - u ) / std
print("norm x=%s, u=%s, std=%s, z=%s, prob left = %s" %  (x, u, std, z, stats.norm.cdf(z)))
