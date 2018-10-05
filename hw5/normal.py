from scipy import stats
import math

probToLeftOfZ = 0.975
z = stats.norm.ppf(probToLeftOfZ)
print("prob left of z %s (right %s) is from zValue = %s" %\
(probToLeftOfZ, 1- probToLeftOfZ, z))

print()
print()

zValue = -1.4670


HYPORTEST = True
CI = True
if HYPORTEST:
   n = 10
   xBar = 84
   populationStd = 9.1
   populationMeanTest = 74.4
   hytestAlpha = 0.01
   print("n = %s, xBar = %s, populationStd = %s, populationMeanTest = %s" %\
         (n, xBar, populationStd, populationMeanTest))
   print("testing H0: u = %s, alpha = %s" % (populationMeanTest, hytestAlpha))
   zValue = (xBar - populationMeanTest) / (populationStd / math.sqrt(n))

probLeft = stats.norm.cdf(zValue)
print("z = %s , lookup prob to the left of this is %s (right %s)" %\
(zValue, probLeft, 1 - probLeft))

if HYPORTEST:
   pVal = 2 * (1-probLeft)
   print("print p-value = %s, %s" % (pVal, "NOT reject" if pVal > hytestAlpha else "Reject"))


   if CI:
      CITwoSideConfident = 1 - hytestAlpha
      probRight = (1 - CITwoSideConfident) / 2
      z = stats.norm.ppf(1-probRight)
      CI_0 = (-z * populationStd / math.sqrt(n)) + xBar
      CI_1 = (z * populationStd / math.sqrt(n)) + xBar
      print("%s%% CI = (%s , %s)" % (CITwoSideConfident*100, CI_0, CI_1))
      if populationMeanTest >= CI_0 and populationMeanTest <= CI_1: 
         print("population mean test %s is within CI. Not Reject" % populationMeanTest)
      else:
         print("population mean test %s is outside CI. Reject" % populationMeanTest)




