"""
House Price Prediction using Ensemble Learning

Workflow:
1. Handle missing values
2. Encode categorical features
3. Split data into training and testing sets
4. Train a Gradient Boosting Regressor
5. Evaluate model performance using R², MAE, and RMSE
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import(r2_score,mean_absolute_error,mean_squared_error)

df=pd.read_csv("data/train.csv")

#filing nan missing values with none 

no_feature=[
    "PoolQC",
    "Alley",
    "Fence",
    "FireplaceQu",
    "GarageType",
    "MasVnrType"
]
for col in no_feature:
    df[col]=df[col].fillna("none")

#filling the numerical missing values

df["LotFrontage"]=df["LotFrontage"].fillna(df["LotFrontage"].median())
df["MasVnrArea"]=df["MasVnrArea"].fillna(0)
df["GarageYrBlt"]=df["GarageYrBlt"].fillna(0)

#one hot encode categorical features

df=pd.get_dummies(df,drop_first=True,dtype=int)

# Split freatures and target

x=df.drop(["SalePrice","Id"],axis=1)
y=df["SalePrice"]

x_train,x_test,y_train,y_test=train_test_split(
    x,y,
    random_state=42,
    test_size=0.2
)

# Train Gradient Boosting model

# Best parameters obtained throught experimentation and tuning
gbr=GradientBoostingRegressor(
    random_state=42,
    n_estimators=500,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8
)

gbr.fit(x_train,y_train)

#Generate predictions

train_pred=gbr.predict(x_train)
test_pred=gbr.predict(x_test)

# Evaluate model

mae=mean_absolute_error(y_test,test_pred)
mse=mean_squared_error(y_test,test_pred)
rmse=np.sqrt(mse)

train_r2 = r2_score(y_train, train_pred)
test_r2 = r2_score(y_test, test_pred)

print(f"Train R² Score: {train_r2:.4f}")
print(f"Test R² Score: {test_r2:.4f}")
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
