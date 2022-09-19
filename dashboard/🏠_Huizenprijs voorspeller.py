import streamlit as st
from lib.predict import regression, decision_tree, neural_network


st.markdown("# Huizenprijs voorspeller 🏠")
st.sidebar.markdown("""
    # Huizenprijs voorspeller 🏠
    TODO
""")

# Show user inputs
in_col1, in_col2, in_col3 = st.columns(3)
with in_col1:
    mo_sold = st.select_slider(
        'Month of sale',
        options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        value=1
    )

with in_col2:
    yr_sold = st.slider('Year of sale', min_value=2010, max_value=2020, step=1, value=2020)

with in_col3:
    living_area = st.slider('Living area (sqft)', min_value=1, max_value=20_000, step=1, value=10_000)

# TODO: determine predicting features
X = [mo_sold, yr_sold, living_area]

def result_markdown(title, result):
    st.markdown(f"""
    {title}\\
    **${"{:.2f}".format(result)}**
    """)


# Show calculated results
out_col1, out_col2, out_col3 = st.columns(3)
with out_col1:
    result_markdown("Regressie model", regression(X))

with out_col2:
    result_markdown("Neural network model", neural_network(X))

with out_col3:
    result_markdown("Decision tree model", decision_tree(X))
