
# Boston Housing Price Prediction

## Project Overview
This project aims to predict house prices in Boston using machine learning techniques. The dataset used for this project is the **Boston Housing Dataset**, which contains various features describing houses in the Boston area. The goal is to develop a machine learning model that can predict the median value of homes based on these features.

## Dataset
The dataset used in this project is the **Boston Housing Dataset**, which consists of 506 samples and 13 feature variables. The features include information such as crime rates, average number of rooms per dwelling, accessibility to highways, and more. The target variable is the **median value of owner-occupied homes** (MEDV), in thousands of dollars.

### Data Preprocessing
- Missing values: The dataset has no missing values, so no imputation was required.
- Feature scaling: Some features were scaled using StandardScaler to improve model performance.
- Train-test split: The data was split into training and testing sets (80%-20%).

## Models Used
Two models were trained:
1. **Linear Regression**: A simple model used as a baseline.
2. **Random Forest Regressor**: A more complex model that performs well with non-linear relationships.

The **Random Forest Regressor** model was selected based on its superior performance (lower RMSE and higher R²) compared to Linear Regression.

## Streamlit Application
The project includes a Streamlit application for users to interact with the model. The app allows users to:
- Explore the dataset and view key statistics.
- Visualize relationships between features and the target variable.
- Input new values and get real-time predictions for house prices.

### App Features:
- **Sidebar Navigation**: Organizes content into sections: Home, Data, Visualizations, Predict, Model Performance.
- **Real-Time Prediction**: Users can input values for the features and receive a predicted house price.
- **Visualizations**: Interactive charts like histograms, scatter plots, and correlation heatmaps.

## Deployment
The Streamlit app was deployed on **Streamlit Cloud** and is accessible via a public URL. The app was also linked to a GitHub repository for version control and collaboration.

## Instructions to Run the App Locally
1. Clone the repository:
    ```bash
    git clone <your-repository-url>
    cd boston_housing_app
    ```

2. Set up a virtual environment:
    ```bash
    python -m venv venv
    # Activate the virtual environment:
    # Windows:
    venv\Scriptsctivate
    # macOS/Linux:
    source venv/bin/activate
    ```

3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```

## Requirements
- Python 3.8+
- `pandas`, `numpy`, `scikit-learn`, `joblib`, `streamlit`, `matplotlib`, `seaborn`, `plotly`

## Challenges Faced
- Ensuring that the app loaded the model and scaler files correctly.
- Designing an intuitive and responsive interface for user input.

## Reflection
This project provided valuable hands-on experience with machine learning model development, data preprocessing, and web application deployment using Streamlit.
