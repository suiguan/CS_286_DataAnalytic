from scipy import stats

#Studnt, n=999, p<0.05, 2-tail, df=n-1=998
#equivalent to Excel TINV(0.05,999)
#stats.t.ppf(1-0.025, df+1)

df = 5
tVal = 2.015
probRight = 1 - stats.t.cdf(tVal, df+1)
print("tValue %s, df %s, prob to the right %s" %\
(tVal, df, probRight))

