from joblib import dump
import numpy as np
import pandas as pd
import math
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


df = pd.read_csv('data/train.csv')

X = df[['Mo Sold', 'Yr Sold', 'Gr Liv Area']].values
y = df['SalePrice'].transpose()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

for model, model_name in [
    [LinearRegression(), 'regression'],
    [DecisionTreeRegressor(), 'decision_tree'],
    [MLPRegressor(), 'neural_network'],
]:
    print(f"Running {model_name}")
    model = model.fit(X_train, y_train)
    dump(model, f"models/{model_name}.joblib")

    score = model.score(X_test, y_test)
    error = math.sqrt(mean_squared_error(y_test, model.predict(X_test)))
    print(f"Score: {'{:.2f}'.format(score)}, mean error: ${'{:.2f}'.format(error)}")
