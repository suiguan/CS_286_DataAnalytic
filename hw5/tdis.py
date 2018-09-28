from scipy import stats
import math

#Studnt, n=999, p<0.05, 2-tail, df=n-1=998
#equivalent to Excel TINV(0.05,999)
#stats.t.ppf(1-0.025, df+1)

if False:
   df = 5
   tVal = 2.015
   probRight = 1 - stats.t.cdf(tVal, df+1)
   print("tValue %s, df %s, prob to the right %s" %\
   (tVal, df, probRight))


HYPORTEST = True
CI = True
if HYPORTEST:
   n = 10
   xBar = 37.20
   xStd = 7.13
   populationMeanTest = 4.13
   df = n - 1
   hytestAlpha = 0.05
   print("n = %s, xBar = %s, xStd = %s, populationMeanTest = %s, df = %s" %\
         (n, xBar, xStd, populationMeanTest, df))
   print("testing u = %s, alpha = %s" % (populationMeanTest, hytestAlpha))

   tValue = (xBar - populationMeanTest) / (xStd / math.sqrt(n))

probLeft = stats.t.cdf(tValue, df+1)
print("t = %s , lookup prob to the left of this is %s (right %s)" %\
(tValue, probLeft, 1 - probLeft))

if HYPORTEST:
   pVal = 2 * (1-probLeft)
   print("print p-value = %s, %s" % (pVal, "NOT Reject" if pVal > hytestAlpha else "Reject"))


   if CI:
      CITwoSideConfident = 0.95
      probRight = (1 - CITwoSideConfident) / 2
      t = stats.t.ppf(1-probRight, df+1)
      CI_0 = (-t * xStd / math.sqrt(n)) + xBar
      CI_1 = (t * xStd / math.sqrt(n)) + xBar
      print("%s%% CI = (%s , %s)" % (CITwoSideConfident*100, CI_0, CI_1))
      if populationMeanTest >= CI_0 and populationMeanTest <= CI_1: 
         print("population mean test %s is within CI. Not Reject" % populationMeanTest)
      else:
         print("population mean test %s is outside CI. Reject" % populationMeanTest)

