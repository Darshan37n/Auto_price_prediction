# Auto Price Prediction

A machine learning project that predicts car prices based on technical specifications, design attributes, and manufacturer details using a complete end-to-end ML pipeline.

## Dataset
| Detail | Info |
|---|---|
| **Source** | Automobile Imports Dataset |
| **Size** | 200 rows × 26 columns |
| **Target Variable** | `price` — Market price of the car (USD) |

## Key Features
| Feature | Type | Description |
|---|---|---|
| `make` | Categorical | Car brand/manufacturer |
| `engine-size` | Numerical | Engine displacement (cc) |
| `curb-weight` | Numerical | Car weight without passengers (lbs) |
| `horsepower` | Numerical | Engine power output |
| `fuel-type` | Categorical | Gas or Diesel |
| `body-style` | Categorical | Sedan, hatchback, convertible, etc. |
| `drive-wheels` | Categorical | FWD, RWD, or 4WD |
| `city-mpg` | Numerical | Fuel efficiency in city (miles/gallon) |
| `price` | Numerical | Target variable |

## Project Workflow
1. Data Cleaning

- Replaced invalid "?" symbols with NaN in numerical columns
- Imputed missing numerical values using median
- Imputed missing categorical values using mode

2. Exploratory Data Analysis (EDA)

- Univariate, Bivariate, and Multivariate analysis
- Engine size, curb weight, and horsepower show strong positive correlation with price
- City/highway MPG shows negative correlation with price
- Luxury brands and rear-wheel-drive cars tend to have higher prices

3. Feature Engineering
-Outlier Handling:
  * Detected using IQR method
  * Applied winsorization (capping) to preserve dataset size

## Hybrid Encoding Strategy:

| Encoding Type | Features Applied To |
|---|---|
| Target Encoding | `make` (high cardinality) |
| Label Encoding | `fuel-type`, `aspiration`, `engine-location` |
| Ordinal Encoding | `num-of-doors`, `num-of-cylinders` |
| One-Hot Encoding | `body-style`, `drive-wheels`, `engine-type`, `fuel-system` |

4. Feature Selection

Removed multicollinear features using correlation threshold of 0.85
Dropped: num-of-cylinders, horsepower, length, wheel-base, bore

5. Modeling

| Model | Notes |
|---|---|
| Linear Regression | Baseline |
| Ridge Regression (L2) | Reduces overfitting |
| Lasso Regression (L1) | Feature sparsity |
| Gradient Boosting | Strong non-linear learning |
| **XGBoost** | Best performance |

## Tools Used

Python — Pandas, NumPy, Matplotlib, Seaborn
Scikit-Learn — Preprocessing, Encoding, Regression Models
XGBoost
