import streamlit as st
import pandas as pd
from joblib import load


def X_from_user_inputs(
    overall_quality=6,
    basement_area=1_000,
    first_floor_area=1_250,
    second_floor_area=350,
    living_area=1_500,
    garage_cars=2,
    garage_area=500,
    lot_frontage=60
):
    return pd.DataFrame(
        {
            'Overall Qual': overall_quality,
            'Total Bsmt SF': basement_area,
            '1st Flr SF': first_floor_area,
            '2nd Flr SF': second_floor_area,
            'Gr Liv Area': living_area,
            'Garage Cars': garage_cars,
            'Garage Area': garage_area,
            'Lot Frontage': lot_frontage,
        },
        index=[0]
    ).iloc[0]


@st.cache(ttl=60 * 60)
def regression(X):
    """Fetch regression model and return prediction for X"""

    reg_model = load('models/regression.joblib')
    return reg_model.predict([X])[0]


@st.cache(ttl=60 * 60)
def decision_tree(X):
    """Fetch decision_tree and return prediction for X"""
    model = load('models/decision_tree.joblib')
    return model.predict([X])[0]


@st.cache(ttl=60 * 60)
def xgboost(X):
    model = load('models/xgboost.joblib')
    return model.predict([X])[0]


@st.cache(ttl=60 * 60)
def neural_network(X):
    model = load('models/neural_network.joblib')
    return model.predict([X])[0]
