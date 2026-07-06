"""
House Price Prediction using ElasticNet Regression

Workflow:
1. Handle missing values
2. Encode categorical features
3. Scale numerical features
4. Apply log transformation to the target
5. Tune ElasticNet using GridSearchCV
6. Evaluate model performance
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet      #works with combination of ridge and lasso
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

#one hot encoding 
df=pd.get_dummies(df,drop_first=True,dtype=int)

#now splting training and scaling
x=df.drop(["SalePrice","Id"],axis=1)
y=np.log(df["SalePrice"])
x_train,x_test,y_train,y_test=train_test_split(
    x,y,
    random_state=42,
    test_size=0.2
)

#scaling
scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(x_test)

#now implementing elastic net as it uses both ridge and lasso by tuning and eliminating the coefficient
enet=ElasticNet(random_state=42,max_iter=10000)
param_grid={
    "alpha":[0.001,0.01,0.1,1,10,100],
    "l1_ratio": [0.2, 0.4, 0.6, 0.8, 1.0]
}
grid=GridSearchCV(
    estimator=enet,
    param_grid=param_grid,
    cv=5,
    scoring="r2"
)
grid.fit(x_train_scaled,y_train)
best_enet=grid.best_estimator_

print("Best Parameters:", grid.best_params_)
print("Best CV Score:", grid.best_score_)

# Predictions in log space
train_pred_log = best_enet.predict(x_train_scaled)
test_pred_log = best_enet.predict(x_test_scaled)

# R² in log space
print("Train R²:", r2_score(y_train, train_pred_log))
print("Test R²:", r2_score(y_test, test_pred_log))

# Convert back to original prices
train_pred = np.exp(train_pred_log)
test_pred = np.exp(test_pred_log)

y_train_actual = np.exp(y_train)
y_test_actual = np.exp(y_test)

# Error metrics in original price units
mae = mean_absolute_error(y_test_actual, test_pred)
rmse = np.sqrt(mean_squared_error(y_test_actual, test_pred))

print("MAE:", mae)
print("RMSE:", rmse)

# Feature selection summary
coef = pd.Series(best_enet.coef_, index=x.columns)
print("Features removed:", (coef == 0).sum())