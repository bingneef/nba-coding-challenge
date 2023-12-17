from joblib import dump
import pandas as pd
import math
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
 # Example project
 # This makes no changes to the code base
 # VS and GIT sync?

BASE_NUMERIC_FEATURES = [
    'Overall Qual',
    'Total Bsmt SF',
    '1st Flr SF',
    '2nd Flr SF',
    'Gr Liv Area',
    'Garage Cars',
    'Garage Area',
    'Lot Frontage',
    'SalePrice'
]

DUMMY_VARIABLE_FEATURES = []


def prep_df(df):
    df['Lot Frontage'] = df['Lot Frontage'].fillna(0)
    return df


def prep_features(df):
    # Base features
    df_2 = df[BASE_NUMERIC_FEATURES]
    df_2 = df_2.fillna(0)

    # Dummy features
    for col in DUMMY_VARIABLE_FEATURES:
        df_2 = df_2.join(
            pd.get_dummies(df[col])
        )

    return df_2


def train_models(df):
    y_col = 'SalePrice'
    X = df.drop(y_col, axis=1).values
    y = df[y_col].transpose()

    print(df[['Total Bsmt SF', '1st Flr SF', '2nd Flr SF', 'Gr Liv Area']].describe())

    for x in df.columns:
        print(x)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    scores = []

    for model, model_name in [
        [LinearRegression(), 'regression'],
        [DecisionTreeRegressor(criterion="squared_error"), 'decision_tree'],
        [XGBRegressor(objective="reg:squarederror"), 'xgboost'],
        [MLPRegressor(), 'neural_network'],
    ]:
        print(f"Running {model_name}")
        model = model.fit(X_train, y_train)
        dump(model, f"models/{model_name}.joblib")

        score = model.score(X_test, y_test)
        error = math.sqrt(mean_squared_error(y_test, model.predict(X_test)))
        print(f"Score: {'{:.3f}'.format(score)}, mean squared error: ${'{:.2f}'.format(error)}")
        print('---')

        scores.append({'name': model_name, 'score': '{:.3f}'.format(score), 'msqe': '{:.2f}'.format(error)})

    df_scores = pd.DataFrame(scores)
    df_scores.to_csv('data/scores.csv')


def main():
    df = pd.read_csv('data/AmesHousing.csv')
    df = prep_df(df)
    df = prep_features(df)
    train_models(df)


if __name__ == '__main__':
    main()