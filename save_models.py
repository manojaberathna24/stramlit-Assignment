# save_models.py
import pickle
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


# Load data
df = pd.read_csv("boston.csv")
X = df.drop(columns=["MEDV"])
y = df["MEDV"]

# Train models
lr = LinearRegression()
rf = RandomForestRegressor(n_estimators=100, random_state=42)
lr.fit(X, y)
rf.fit(X, y)

# Save models
with open("lr_model.pkl", "wb") as f1, open("rf_model.pkl", "wb") as f2:
    pickle.dump(lr, f1)
    pickle.dump(rf, f2)