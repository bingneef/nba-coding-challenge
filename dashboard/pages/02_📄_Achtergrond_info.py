import streamlit as st
import pandas as pd
import string


st.set_page_config(
    page_title="Achtergrond info 📄",
    layout="wide"
)


st.markdown("# Achtergrond info 📄")

tab1, tab2, tab3, tab4 = st.tabs(["Model performance", "Resultaten data exploration", "Model code", "Bron bestand"])

with tab1:
    df = pd.read_csv('data/scores.csv', usecols=['name', 'score', 'msqe'])

    st.markdown('# Model performance 💪')
    st.markdown('De scores van de verschillende modellen met de Mean Squared Error in dollars.')

    cols = st.columns(4)
    for index, row in df.iterrows():
        with cols[index % 4]:
            st.metric(
                label=string.capwords(row['name'].replace('_', ' ')),
                value='{:0.3f}'.format(row['score']),
                delta=f"$ {'{:,.2f}'.format(row['msqe'])}",
                delta_color='inverse')

with tab2:
    with open("data/data-exploration.html", 'r', encoding='utf-8') as file:
        source_code = file.read()

    st.components.v1.html(source_code, height=1000, scrolling=True)

with tab3:
    with open('model.py', mode='r') as file:
        model_code = file.read()
    st.markdown("Met de onderstaande code is het model getraind.")
    st.code(model_code, language='python')

with tab4:
    st.markdown("Download hier het CSV bron bestand wat voor de modellen gebruikt is.")
    with open('data/AmesHousing.csv', mode='rb') as file:
        csv_data = file.read()
    st.download_button('Download csv', data=csv_data, file_name='bron-bestand.csv')

    st.markdown("Bij de kolommen in de dataset horen de volgende beschrijvingen.")
    feature_description = ("\n"
                           "- **SalePrice**: - the property's sale price in dollars.\n"
                           "- **MSSubClass**: The building class\n"
                           "- **MSZoning**: The general zoning classification\n"
                           "- **LotFrontage**: Linear feet of street connected to property\n"
                           "- **LotArea**: Lot size in square feet\n"
                           "- **Street**: Type of road access\n"
                           "- **Alley**: Type of alley access\n"
                           "- **LotShape**: General shape of property\n"
                           "- **LandContour**: Flatness of the property\n"
                           "- **Utilities**: Type of utilities available\n"
                           "- **LotConfig**: Lot configuration\n"
                           "- **LandSlope**: Slope of property\n"
                           "- **Neighborhood**: Physical locations within Ames city limits\n"
                           "- **Condition1**: Proximity to main road or railroad\n"
                           "- **Condition2**: Proximity to main road or railroad (if a second is present)\n"
                           "- **BldgType**: Type of dwelling\n"
                           "- **HouseStyle**: Style of dwelling\n"
                           "- **OverallQual**: Overall material and finish quality\n"
                           "- **OverallCond**: Overall condition rating\n"
                           "- **YearBuilt**: Original construction date\n"
                           "- **YearRemodAdd**: Remodel date\n"
                           "- **RoofStyle**: Type of roof\n"
                           "- **RoofMatl**: Roof material\n"
                           "- **Exterior1st**: Exterior covering on house\n"
                           "- **Exterior2nd**: Exterior covering on house (if more than one material)\n"
                           "- **MasVnrType**: Masonry veneer type\n"
                           "- **MasVnrArea**: Masonry veneer area in square feet\n"
                           "- **ExterQual**: Exterior material quality\n"
                           "- **ExterCond**: Present condition of the material on the exterior\n"
                           "- **Foundation**: Type of foundation\n"
                           "- **BsmtQual**: Height of the basement\n"
                           "- **BsmtCond**: General condition of the basement\n"
                           "- **BsmtExposure**: Walkout or garden level basement walls\n"
                           "- **BsmtFinType1**: Quality of basement finished area\n"
                           "- **BsmtFinSF1**: Type 1 finished square feet\n"
                           "- **BsmtFinType2**: Quality of second finished area (if present)\n"
                           "- **BsmtFinSF2**: Type 2 finished square feet\n"
                           "- **BsmtUnfSF**: Unfinished square feet of basement area\n"
                           "- **TotalBsmtSF**: Total square feet of basement area\n"
                           "- **Heating**: Type of heating\n"
                           "- **HeatingQC**: Heating quality and condition\n"
                           "- **CentralAir**: Central air conditioning\n"
                           "- **Electrical**: Electrical system\n"
                           "- **1**stFlrSF: First Floor square feet\n"
                           "- **2**ndFlrSF: Second floor square feet\n"
                           "- **LowQualFinSF**: Low quality finished square feet (all floors)\n"
                           "- **GrLivArea**: Above grade (ground) living area square feet\n"
                           "- **BsmtFullBath**: Basement full bathrooms\n"
                           "- **BsmtHalfBath**: Basement half bathrooms\n"
                           "- **FullBath**: Full bathrooms above grade\n"
                           "- **HalfBath**: Half baths above grade\n"
                           "- **Bedroom**: Number of bedrooms above basement level\n"
                           "- **Kitchen**: Number of kitchens\n"
                           "- **KitchenQual**: Kitchen quality\n"
                           "- **TotRmsAbvGrd**: Total rooms above grade (does not include bathrooms)\n"
                           "- **Functional**: Home functionality rating\n"
                           "- **Fireplaces**: Number of fireplaces\n"
                           "- **FireplaceQu**: Fireplace quality\n"
                           "- **GarageType**: Garage location\n"
                           "- **GarageYrBlt**: Year garage was built\n"
                           "- **GarageFinish**: Interior finish of the garage\n"
                           "- **GarageCars**: Size of garage in car capacity\n"
                           "- **GarageArea**: Size of garage in square feet\n"
                           "- **GarageQual**: Garage quality\n"
                           "- **GarageCond**: Garage condition\n"
                           "- **PavedDrive**: Paved driveway\n"
                           "- **WoodDeckSF**: Wood deck area in square feet\n"
                           "- **OpenPorchSF**: Open porch area in square feet\n"
                           "- **EnclosedPorch**: Enclosed porch area in square feet\n"
                           "- **3**SsnPorch: Three season porch area in square feet\n"
                           "- **ScreenPorch**: Screen porch area in square feet\n"
                           "- **PoolArea**: Pool area in square feet\n"
                           "- **PoolQC**: Pool quality\n"
                           "- **Fence**: Fence quality\n"
                           "- **MiscFeature**: Miscellaneous feature not covered in other categories\n"
                           "- **MiscVal**: $Value of miscellaneous feature\n"
                           "- **MoSold**: Month Sold\n"
                           "- **YrSold**: Year Sold\n"
                           "- **SaleType**: Type of sale\n"
                           "- **SaleCondition**: Condition of sale\n")
    st.markdown(feature_description)