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
    area = st.number_input('Area (sqft)')

# FIXME: determine predicting features
X = [area,area]

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
