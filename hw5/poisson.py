from scipy.stats import poisson

poi = poisson(2.75)
s = 0
for i in range(0, 6):
   p = poi.pmf(i)
   s += p
   print("%s = %s" % (i, p))
print("sum = ", s)
print("1-sum = ", 1-s)
