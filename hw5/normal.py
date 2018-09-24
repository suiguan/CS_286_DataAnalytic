from scipy import stats

probToLeftOfZ = 0.975
z = stats.norm.ppf(probToLeftOfZ)
print("prob left of z %s (right %s) is from zValue = %s" %\
(probToLeftOfZ, 1- probToLeftOfZ, z))

print()
print()

zValue = -1.4670
probLeft = stats.norm.cdf(zValue)
print("z = %s , lookup prob to the left of this is %s (right %s)" %\
(zValue, probLeft, 1 - probLeft))

