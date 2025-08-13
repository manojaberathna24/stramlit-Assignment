import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from PIL import Image
import cv2
import io
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression




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
    h1 { text-align: center; } 
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
        ["Data Exploration", "Visualisations", "Model Prediction", "Model Performance", "Image Processing"]
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
                time.sleep(1)  
                input_df = pd.DataFrame([input_data])
                prediction = model.predict(input_df)[0]
                st.success(f" Predicted House Price: ${prediction * 1000:,.2f}")
            except Exception as e:
                st.error(f"Prediction error: {e}")

# =============================
# MODEL PERFORMANCE
elif menu == "Model Performance":
    st.title(" Model Performance (Classification Style)")
    
    # Convert to classification by binning prices
    df['PRICE_CATEGORY'] = pd.cut(df['MEDV'],
                                 bins=[0, 15, 25, 50],  
                                 labels=['Low', 'Medium', 'High'])
    
    X = df.drop(columns=['MEDV', 'PRICE_CATEGORY'])
    y = df['PRICE_CATEGORY']
    
    # Train classification model
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    # Display classification metrics
    st.write(f"**Accuracy:** {accuracy_score(y_test, y_pred):.3f}")
    
    # Show metrics for each class
    report = classification_report(y_test, y_pred, output_dict=True)
    st.write(f"**Precision (Weighted Avg):** {report['weighted avg']['precision']:.3f}")
    st.write(f"**Recall (Weighted Avg):** {report['weighted avg']['recall']:.3f}")
    st.write(f"**F1 Score (Weighted Avg):** {report['weighted avg']['f1-score']:.3f}")
    
    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred))
    
    # Confusion matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
                xticklabels=clf.classes_,
                yticklabels=clf.classes_)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)
    
    
    # =============================
# IMAGE PROCESSING
# =============================
elif menu == "Image Processing":
    st.title(" House Image Artistry")
    st.write("Transform house images into artistic styles using AI processing.")
    
    # Create two columns for layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Processing Options")
        processing_type = st.radio(
            "Select art style:",
            ["Pencil Sketch", "Watercolor", "Black & White"],
            index=0,
            help="Choose the artistic style you want to apply"
        )
        
        uploaded_file = st.file_uploader(
            "Upload house photo", 
            type=["jpg", "jpeg", "png"],
            help="Select an image of a house to transform"
        )
        
        # Additional settings
        with st.expander("Advanced Settings"):
            if processing_type == "Pencil Sketch":
                blur_strength = st.slider("Sketch Intensity", 5, 50, 21, 2)
            elif processing_type == "Watercolor":
                watercolor_strength = st.slider("Brush Stroke Size", 10, 100, 60)
    
    with col2:
        if uploaded_file is not None:
            st.subheader("Artistic Transformation")
            original_image = Image.open(uploaded_file)
            
            # Display original and processed images side by side
            if st.button(f"Create {processing_type} Art", type="primary"):
                with st.spinner(f"Working on your {processing_type.lower()} masterpiece..."):
                    try:
                        img_array = np.array(original_image)
                        
                        if processing_type == "Pencil Sketch":
                            gray_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                            inverted = 255 - gray_image
                            blurred = cv2.GaussianBlur(inverted, (blur_strength, blur_strength), 0)
                            result = cv2.divide(gray_image, 255 - blurred, scale=256.0)
                            
                        elif processing_type == "Watercolor":
                            resized = cv2.resize(img_array, (800, 800))
                            result = cv2.stylization(
                                resized, 
                                sigma_s=watercolor_strength, 
                                sigma_r=0.6
                            )
                            
                        elif processing_type == "Black & White":
                            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                            _, result = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
                        
                        result_image = Image.fromarray(result)
                        
                        # Display comparison
                        st.image(
                            [original_image, result_image],
                            caption=["Original Image", f"{processing_type} Version"],
                            width=300
                        )
                        
                        # Download button
                        buf = io.BytesIO()
                        result_image.save(buf, format="PNG")
                        st.download_button(
                            "Download Artwork",
                            buf.getvalue(),
                            f"house_art_{processing_type.lower().replace(' ', '_')}.png",
                            "image/png"
                        )
                        
                    except Exception as e:
                        st.error(f"Processing error: {str(e)}")
                        st.info("Please ensure OpenCV is installed: pip install opencv-python")
        
        else:
            st.info("Please upload an image to begin processing")
            st.image("https://cdn.pixabay.com/photo/2016/11/18/17/46/house-1836070_640.jpg",
                   caption="Example House Image",
                   use_column_width=True)
