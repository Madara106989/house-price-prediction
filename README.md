# House Price Prediction using Ensemble learning

## Project Overview

This project predicts residential house prices using the House Prices – Advanced Regression Techniques dataset from Kaggle.

The project demonstrates an end-to-end machine learning workflow including domain-driven missing value handling, feature engineering, preprocessing, model comparison, hyperparameter tuning, and ensemble learning techniques for regression.

---

## Repository Highlights

- ✔ End-to-end regression pipeline
- ✔ Domain-driven missing value handling
- ✔ Comparison of linear and ensemble methods
- ✔ Hyperparameter tuning using GridSearchCV
- ✔ Final Gradient Boosting model achieving Test R² of 0.915

---

## Objective

Predict the **SalePrice** of residential houses based on their structural and neighborhood characteristics.

---

## Dataset

- **Source:** Kaggle - House Prices: Advanced Regression Techniques
- **Rows:** 1,460
- **Features:** 81
- **Target Variable:** `SalePrice`

---

## Technologies Used

- Python
- NumPy
- Pandas
- Scikit-Learn

---

## Project Workflow

### 1. Data Preprocessing

- Loaded the dataset
- Identified missing values
- Applied domain-driven missing value handling

### Missing Value Strategy

Instead of filling missing values blindly, each feature was analyzed based on its meaning.

Examples:

| Feature | Strategy | Reason |
|---------|----------|--------|
| PoolQC | "none" | House has no pool |
| Alley | "none" | No alley access |
| Fence | "none" | No fence |
| FireplaceQu | "none" | No fireplace |
| GarageType | "none" | No garage |
| MasVnrType | "none" | No masonry veneer type |
| LotFrontage | Median | Every house has frontage |
| MasVnrArea | 0 | No masonry veneer area |
| GarageYrBlt | 0 | No garage |

---

### 2. Feature Engineering

- One-Hot Encoding for categorical variables
- `drop_first=True` to avoid the Dummy Variable Trap
- Removed identifier column (`Id`) before training

---

### 3. Train-Test Split

- 80% Training Data
- 20% Testing Data

---

### 4. Feature Scaling

Applied **StandardScaler** after train-test split to prevent data leakage.

---

### 5. Model Selection

The following models were explored during experimentation:

- Linear Regression
- Ridge Regression
- Lasso Regression
- Elastic Net Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor

After comparing performance across models, Gradient Boosting Regressor achieved the best test performance and was selected as the final model.

---

### 6. Hyperparameter Tuning

GridSearchCV with 5-fold cross-validation was used for hyperparameter optimization.

Examples of tuned parameters:

Elastic Net:
- alpha
- l1_ratio

Gradient Boosting:
- learning_rate
- n_estimators
- max_depth
- subsample

---

### 7. Challenges Faced

During development several practical machine learning challenges were encountered:

- Handling missing values based on feature semantics instead of using a single imputation strategy.
- Diagnosing multicollinearity using Variance Inflation Factor (VIF).
- Understanding coefficient shrinkage in Ridge and feature selection in Lasso.
- Comparing multiple regularized regression models.
- Experimenting with logarithmic target transformation and evaluating its impact on model performance.
- Understanding the bias-variance tradeoff in tree-based ensemble methods.
- Comparing linear and nonlinear models for tabular data.

---

## Model Comparison

| Model | Train R² | Test R² |
|-------|----------:|---------:|
| Elastic Net | 0.887 | 0.851 |
| Decision Tree | 0.875 | 0.804 |
| Random Forest | 0.970 | 0.892 |
| Gradient Boosting | 0.999 | 0.915 |

Gradient Boosting achieved the best performance and was selected as the final model.

---

## Final Model Performance

| Metric | Score |
|---------|-------:|
| Training R² | 0.9994 |
| Testing R² | 0.9154 |
| MAE | 16,004 |
| RMSE | 25,477 |

---

## Key Learning Outcomes

During this project I learned:

- Domain-driven missing value handling
- One-Hot Encoding
- Avoiding the Dummy Variable Trap
- Feature scaling
- Train/Test split best practices
- Data leakage prevention
- Multicollinearity
- Ridge Regression
- Lasso Regression
- Elastic Net Regression
- Hyperparameter tuning using GridSearchCV
- Model evaluation using R²
- Target Variable Transformation
- Log Transformation
- Understanding skewed distributions
- Decision Trees
- Bagging
- Random Forest
- Gradient Boosting
- Ensemble Learning
- Bias-Variance Tradeoff
- Cross Validation

---

## Project Structure

```
House-Price-Prediction/
│
├── data/
│   └── train.csv
│
├── house_price_prediction.py
├── README.md
├── requirements.txt
└── .gitignore
```

---

## How to Run

Clone the repository

```bash
git clone https://github.com/Madara106989/house-price-prediction.git
```

Move into the project folder

```bash
cd house-price-prediction
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the project

```bash
python house_price_prediction.py
```

---

## Future Improvements

Future versions of this project may include:

- XGBoost
- LightGBM
- CatBoost
- Feature Importance Visualization
- Model Explainability using SHAP
- Model Deployment using Flask or Streamlit

---

## Results Summary

Multiple regression and ensemble models were explored and objectively compared throughout this project.

Among all models, Gradient Boosting Regressor achieved the best performance:

- Training R²: 0.9994
- Testing R²: 0.9154
- MAE: 16,004
- RMSE: 25,477

The project demonstrates how systematic preprocessing, feature engineering, hyperparameter tuning, and ensemble learning can significantly improve predictive performance on structured tabular datasets.

---

## Key Takeaways

- Started with simple linear models and established a baseline.
- Explored regularization techniques to handle multicollinearity.
- Investigated tree-based methods and ensemble learning.
- Compared multiple algorithms objectively using train and test metrics.
- Selected the final model based on generalization performance rather than training accuracy.

---

##  Author

Created as part of my Data Science learning journey, focusing on building a strong understanding of machine learning concepts rather than simply applying models.