import sys, os
import numpy as np
import scipy as sp
import pandas as pd
from sklearn.preprocessing import Imputer
import matplotlib.pyplot as plt

data_file = "Guan_data.csv"
df = pd.read_csv(data_file)

df.plot.scatter(x='x2', y='resp')
plt.show()
