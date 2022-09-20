import streamlit as st
import pandas as pd
from joblib import load


def X_from_user_inputs(month_sold, year_sold, living_area):
    return pd.DataFrame(
        {
            'Mo Sold': month_sold, 
            'Yr Sold': year_sold,
            'Gr Liv Area': living_area
        }, 
        index=[0]
    ).iloc[0]

@st.cache
def regression(X):
    """Fetch regression model and return prediction for X"""

    reg_model = load('models/regression.joblib') 
    return reg_model.predict([X])[0]


# TODO: Fix implementation
@st.cache
def decision_tree(X):
    """Fetch decision_tree and return prediction for X"""
    model = load('models/decision_tree.joblib') 
    return model.predict([X])[0]


# TODO: Fix implementation
@st.cache
def neural_network(X):
    model = load('models/neural_network.joblib') 
    return model.predict([X])[0]
