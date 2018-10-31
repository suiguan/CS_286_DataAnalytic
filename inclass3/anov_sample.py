from scipy.stats import f as fDis


#samples = [
#   (21, 2.63, 0.496),
#   (16, 3.03, 0.523),
#   (23, 2.88, 0.498),
#]
#each sample is (n, Xmean, Std)
samples = [
   (73, 6.22, 1.62),
   (105, 5.81, 1.43),
   (240, 5.77, 1.24),
   (1080, 5.47, 1.31),
]

#get n, k
k = len(samples)
n = 0
for ni, xi, s1 in samples: n += ni

#within group
sw2 = 0 
for ni, xi, s1 in samples: sw2 += ((ni-1)*s1*s1)
sw2 /= (n-k)

#between group
xBar = 0
for ni, xi, s1 in samples: xBar += (ni*xi)
xBar /= n
sb2 = 0 
for ni, xi, s1 in samples: sb2 += (ni*(xi-xBar)*(xi-xBar))
sb2 /= (k-1)

f = sb2 / sw2
df1 = k-1
df2 = n-k
pVal = 1-fDis.cdf(f, df1, df2)
print("df1 = %s, df2 = %s, sb2 = %s, sw2 = %s, f = %s, p_value = %s" % (df1, df2, sb2, sw2, f, pVal)) 
