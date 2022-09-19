import streamlit as st
import pandas as pd
from lib.predict import regression, decision_tree, neural_network


st.markdown("# Huizenprijs voorspeller 🏠")
st.sidebar.markdown("""
    # Huizenprijs voorspeller 🏠
    TODO
""")

def month_number_from_month(month):
    return {
        "januari": 1,
        "februari": 2,
        "maart": 3,
        "april": 4,
        "mei": 5,
        "juni": 6,
        "juli": 7,
        "augustus": 8,
        "september": 9,
        "oktober": 10,
        "november": 11,
        "december": 12
    }[month]


def result_markdown(title, result):
    st.markdown(f"""
    {title}\\
    **${"{:.2f}".format(result)}**
    """)


# Show user inputs
col1, col2, col3 = st.columns(3)
with col1:
    mo_sold = st.selectbox(
        'Verkoop maand',
        ("januari", "februari", "maart", "april", "mei", "juni", "juli", "augustus", "september", "oktober", "november", "december"),
        index=0
    )

with col2:
    yr_sold = st.slider('Verkoop jaar', min_value=2010, max_value=2020, step=1, value=2020)

with col3:
    living_area = st.slider('Woonoppervlak (sqft)', min_value=1, max_value=7_500, step=1, value=1_500)


# TODO: determine predicting features
X = [
    month_number_from_month(mo_sold), 
    yr_sold, 
    living_area
]


# Show calculated results
col1, col2, col3 = st.columns(3)
with col1:
    result_markdown("Regressie model", regression(X))

with col2:
    result_markdown("Decision tree model", decision_tree(X))

with col3:
    result_markdown("Neural network model", neural_network(X))
