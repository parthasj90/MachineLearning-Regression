import pandas as pd
import numpy as np
import matplotlib as plt

dataset = pd.read_csv('housing.csv')
X = dataset.iloc[:,:-1]
y = dataset.iloc[:,-1]