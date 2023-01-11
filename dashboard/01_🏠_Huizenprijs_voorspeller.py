import streamlit as st
from lib.predict import regression, decision_tree, neural_network, xgboost, X_from_user_inputs


st.set_page_config(
    page_title="Huizenprijs voorspeller 🏠",
    layout="wide"
)


st.markdown("# Huizenprijs voorspeller 🏠")
st.markdown("## Invoervelden")
extended_input = st.checkbox("Alle velden tonen", value=False)

# Show user inputs
container = st.container()
container.markdown("### Algemeen")
col1, col2, col3 = st.columns(3)
with col1:
    overall_quality = st.slider('Wat is de staat van het huis?', min_value=1, max_value=10, step=1, value=6)

if extended_input is True:
    container = st.container()
    container.markdown("### De buitenkant")
    col1, col2, col3 = st.columns(3)
    with col1:
        garage_area = st.slider("Hoe groot is de garage? (sqft)", min_value=0, max_value=5_000, step=1, value=500)

    with col2:
        garage_cars = st.slider("Hoeveel auto's passen in de garage?", min_value=0, max_value=10, step=1, value=2)

    with col3:
        lot_frontage = st.slider(
            "Hoeveel grenst het aan de weg? (ft)",
            min_value=0,
            max_value=1_000,
            step=1,
            value=60
        )
else:
    garage_area = 500
    garage_cars = 2
    lot_frontage = 60

container = st.container()
container.markdown("### De binnenkant")

col1, col2, col3 = st.columns(3)
with col1:
    living_area = st.slider(
        "Hoe groot is het totale woonoppervlak? (sqft)",
        min_value=0,
        max_value=10_000,
        step=1,
        value=1_500
    )

if extended_input is True:
    col1, col2, col3 = st.columns(3)
    with col1:
        basement_area = st.slider(
            "Hoe groot is de kelder? (sqft)",
            min_value=0,
            max_value=10_000,
            step=1,
            value=1_000
        )

    with col2:
        first_floor_area = st.slider(
            "Hoe groot is de begane grond? (sqft)",
            min_value=0,
            max_value=10_000,
            step=1,
            value=1_250
        )

    with col3:
        second_floor_area = st.slider(
            "Hoe groot is de eerste verdieping? (sqft)",
            min_value=0,
            max_value=3_000,
            step=1,
            value=350
        )

else:
    basement_area = 1_000
    first_floor_area = 1_250
    second_floor_area = 350


X = X_from_user_inputs(
    overall_quality=overall_quality,
    garage_cars=garage_cars,
    garage_area=garage_area,
    lot_frontage=lot_frontage,
    basement_area=basement_area,
    first_floor_area=first_floor_area,
    second_floor_area=second_floor_area,
    living_area=living_area
)

st.markdown("## Resultaten")

def format_currency(amount):
    return f"${'{:,.2f}'.format(amount)}"


# Show calculated results
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(
        label='Regression',
        value=format_currency(regression(X))
    )

with col2:
    st.metric(
        label='Decision Tree',
        value=format_currency(decision_tree(X))
    )

with col3:
    st.metric(
        label='Xgboost',
        value=format_currency(xgboost(X))
    )

with col4:
    st.metric(
        label='Neural Network',
        value=format_currency(neural_network(X))
    )
