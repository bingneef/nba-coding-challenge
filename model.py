from joblib import dump
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn import datasets
import pandas as pd

df = pd.read_csv('data/train.csv')
X = df[['Mo Sold', 'Yr Sold', 'Gr Liv Area']]
y = df['SalePrice'].transpose()

print("Running regression")
regression = LinearRegression().fit(X, y)
dump(regression, 'models/regression.joblib') 

print("Running decision tree")
decision_tree = DecisionTreeRegressor().fit(X, y)
dump(decision_tree, 'models/decision_tree.joblib') 

print("Running neural network")
neural_network = MLPRegressor().fit(X, y)
dump(neural_network, 'models/neural_network.joblib') 