from scipy import stats
import math

sameVar = True 
testRelation = '='
n1 = 8 
n2 = 10 
x1 = 87.38
x2 = 90.14 
s1 = 4.56
s2 = 4.58
alpha = 0.05
diffTest = 0

df = n1 + n2 - 2
sp2 = ((n1-1)*(s1**2) + (n2-1)*(s2**2)) / df 
t = ((x1 - x2) - (diffTest)) / math.sqrt(sp2*((1.0/n1) + (1.0/n2)))

probLeft = stats.t.cdf(t, df)
print("%s = %s , df = %s, lookup prob to the left of this is %s (right %s)" %\
   ("t-dis", t, df, probLeft, 1 - probLeft))

if testRelation == "=": 
   if probLeft >= 0.5: pVal = 2 * (1-probLeft)
   else: pVal = 2 * probLeft
elif testRelation == ">=": pVal = probLeft
elif testRelation == "<=": pVal = 1-probLeft
else: raise Exception("Invalid test relation %s" % testRelation)

print("p-value = %s, %s" % (pVal, "NOT Reject" if pVal >= alpha else "Reject"))

