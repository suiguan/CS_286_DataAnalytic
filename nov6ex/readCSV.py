import sys, os
import numpy as np
import scipy as sp
import pandas as pd
from sklearn.preprocessing import Imputer

data_file = "Guan_data.csv"
df = pd.read_csv(data_file)

print("Descriptive statistics:")
print(df.describe())
print()
print("SE of the mean:")
print(df.sem())
