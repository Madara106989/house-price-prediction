# House Price Prediction using Elastic Net Regression

## Project Overview

This project predicts residential house prices using the **House Prices – Advanced Regression Techniques** dataset from Kaggle.

Rather than focusing only on building a predictive model, the project emphasizes a complete machine learning workflow, including data preprocessing, missing value handling, feature engineering, regularization, hyperparameter tuning, and model evaluation.

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

The following regression models were explored during the learning process:

- Linear Regression
- Ridge Regression
- Lasso Regression
- Elastic Net Regression

Elastic Net was selected as the final model because it combines:

- Ridge Regression (reduces multicollinearity)
- Lasso Regression (performs feature selection)

---

### 6. Hyperparameter Tuning

Used **GridSearchCV** with 5-fold cross-validation.

Parameters tuned:

- alpha
- l1_ratio

---

## Final Model Performance

| Metric | Score |
|---------|-------|
| Best Cross Validation R² | 0.802 |
| Training R² | 0.887 |
| Testing R² | 0.851 |
| Features Removed | 2 |

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
git clone https://github.com/your-username/house-price-prediction.git
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

- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting
- XGBoost
- Feature Importance Visualization
- Model Deployment using Flask or Streamlit

---

##  Author

Created as part of my Data Science learning journey, focusing on building a strong understanding of machine learning concepts rather than simply applying models.