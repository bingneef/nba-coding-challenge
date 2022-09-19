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

for model, model_name in [
    [LinearRegression(), 'regression'],
    [DecisionTreeRegressor(), 'decision_tree'],
    [MLPRegressor(), 'regrneural_networkession'],
]:
    print(f"Running {model_name}")
    dump(model.fit(X, y), f"models/{model_name}.joblib") 