import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import time





# Load model and dataset

@st.cache_resource
def load_model():
    with open("best_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

@st.cache_data
def load_data():
    return pd.read_csv("boston.csv")

model = load_model()
df = load_data()


# Page Configuration

st.set_page_config(page_title="  House Price Predictor", layout="wide")



st.markdown(
    """
    <style>
    .main { background-color: #72F3C6; }
    h1, h2, h3 { color: #333333; }
    </style>
    """,
    unsafe_allow_html=True
)

# =============================
# App Title & Sidebar
# =============================
st.title("  House Price Predictor")
st.markdown("""
Welcome!  
This interactive web application predicts house prices in Boston based on various property features such as crime rate, average number of rooms, and proximity to schools. It utilizes machine learning models, specifically **Linear Regression** and **Random Forest**, to provide an estimated price based on user inputs.
""")

# Navigation with image below
with st.sidebar:
    menu = st.radio(
        "📌 Navigation",
        ["Data Exploration", "Visualisations", "Model Prediction", "Model Performance"]
    )
    
    
    # Add small house photo at the bottom of the sidebar
    st.markdown('<div class="sidebar-image">', unsafe_allow_html=True)
    st.image("https://cdn.pixabay.com/photo/2016/11/18/17/46/house-1836070_640.jpg", 
             width=150, 
             caption="House Price Analysis")
    st.markdown('</div>', unsafe_allow_html=True)





# =============================
# DATA EXPLORATION
# =============================
if menu == "Data Exploration":
    st.header(" Dataset Overview")
    st.info("Here is  the dataset and  see specific records.")

    st.write("**Shape:**", df.shape)
    st.write("**Columns:**", list(df.columns))
    st.write("**Data Types:**")
    st.write(df.dtypes)

    st.subheader(" Sample Data")
    st.dataframe(df.head())

    st.subheader(" Interactive Filtering")
    col_filter = st.selectbox("Select column to filter", df.columns)
    unique_vals = df[col_filter].unique()
    val_filter = st.multiselect("Select values", unique_vals)
    if val_filter:
        st.write(df[df[col_filter].isin(val_filter)])

# =============================
# VISUALISATIONS
# =============================
elif menu == "Visualisations":
    st.header(" Data Visualisations")
    st.info("Choose a chart type to better understand ")

    chart_type = st.selectbox(
        "Choose chart type",
        ["Correlation Heatmap", "Histogram", "Scatter Plot"]
    )

    if chart_type == "Correlation Heatmap":
        with st.spinner("Generating heatmap..."):
            time.sleep(1)  # Simulate long process
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)

    elif chart_type == "Histogram":
        num_col = st.selectbox("Select numeric column", df.select_dtypes(include=np.number).columns)
        fig, ax = plt.subplots()
        sns.histplot(df[num_col], kde=True, ax=ax)
        st.pyplot(fig)

    elif chart_type == "Scatter Plot":
        num_cols = df.select_dtypes(include=np.number).columns
        x_axis = st.selectbox("X-axis", num_cols)
        y_axis = st.selectbox("Y-axis", num_cols)
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x=x_axis, y=y_axis, ax=ax)
        st.pyplot(fig)

# =============================
# MODEL PREDICTION
# =============================
elif menu == "Model Prediction":
    st.header(" Predict House Price")
    st.info("Enter property details below to get an estimated price.")

    friendly_labels = {
        "CRIM": ("Crime Rate", "Number of crimes per capita in the area"),
        "ZN": ("Residential Land %", "Percentage of residential land zoned for large lots"),
        "INDUS": ("Business Land %", "Proportion of land used for business purposes"),
        "CHAS": ("Near River (0 = No, 1 = Yes)", "Is the property near the Charles River?"),
        "NOX": ("Air Pollution Level", "Nitric oxide concentration (ppm)"),
        "RM": ("Avg. Rooms per House", "Average number of rooms per dwelling"),
        "AGE": ("Old Houses %", "Proportion of houses built before 1940"),
        "DIS": ("Distance to Jobs", "Weighted distance to employment centers"),
        "RAD": ("Highway Access", "Index of accessibility to radial highways"),
        "TAX": ("Property Tax Rate", "Full-value property-tax rate per $10,000"),
        "PTRATIO": ("Student-Teacher Ratio", "Average student-teacher ratio in schools"),
        "B": ("Diversity Score", "1000(Bk - 0.63)^2, where Bk is proportion of Black residents"),
        "LSTAT": ("Low-Income %", "Percentage of lower status population")
    }

    input_data = {}
    for col in df.drop(columns=["MEDV"]).columns:
        label, help_text = friendly_labels.get(col, (col, ""))
        if df[col].dtype in [np.float64, np.int64]:
            input_data[col] = st.slider(
                label,
                float(df[col].min()),
                float(df[col].max()),
                float(df[col].mean()),
                help=help_text
            )
        else:
            input_data[col] = st.selectbox(label, df[col].unique(), help=help_text)

    if st.button("Predict"):
        with st.spinner("Calculating prediction..."):
            try:
                time.sleep(1)  # Simulate processing time
                input_df = pd.DataFrame([input_data])
                prediction = model.predict(input_df)[0]
                st.success(f" Predicted House Price: ${prediction * 1000:,.2f}")
            except Exception as e:
                st.error(f"Prediction error: {e}")

# =============================
# =============================
# MODEL PERFORMANCE
# =============================
