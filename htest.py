from scipy import stats
import math

useT = True #True #set it to True if population std is unknown
n = 144 
xBar = 104.736111 
std = 15.604368 
hytestAlpha = 0.05
testRelation = "=" 
populationMeanTest = 100 

df = n - 1
print("n = %s, xBar = %s, std = %s, populationMeanTest = %s, df = %s" %\
      (n, xBar, std, populationMeanTest, df))
print("testing H0: u %s %s, alpha = %s" % (testRelation, populationMeanTest, hytestAlpha))

cValue = (xBar - populationMeanTest) / (std / math.sqrt(n))
if useT: probLeft = stats.t.cdf(cValue, df)
else: probLeft = stats.norm.cdf(cValue)
print("%s = %s , lookup prob to the left of this is %s (right %s)" %\
   ("t-dis" if useT else "z-dis", cValue, probLeft, 1 - probLeft))

if testRelation == "=": 
   if probLeft >= 0.5: pVal = 2 * (1-probLeft)
   else: pVal = 2 * probLeft
elif testRelation == ">=": pVal = probLeft
elif testRelation == "<=": pVal = 1-probLeft
else: raise Exception("Invalid test relation %s" % testRelation)

print("p-value = %s, %s" % (pVal, "NOT Reject" if pVal >= hytestAlpha else "Reject"))

