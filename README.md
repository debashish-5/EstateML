# EstateML : Housing Price Prediction

This project builds a complete machine learning pipeline to predict median house values in California using a housing dataset. It includes exploratory data analysis, preprocessing with pipelines, model training using RandomForestRegressor, and prediction output saved as CSV.

---

## 📁 Project File Overview

| File Name                    | Purpose                                                                  |
|-----------------------------|--------------------------------------------------------------------------|
| `01_Analyzing_the_data.ipynb`   | Explore and understand data distribution, summary, and statistics.    |
| `02_Find_best_ ML Algorithm.ipynb` | Compares ML models like Linear Regression, Decision Trees, etc.     |
| `03_Visualizing_the_data.ipynb` | Data visualization to understand trends and correlations.             |
| `04_ML_(Final Part).ipynb`      | Final pipeline creation and model saving.                             |
| `ML(For - User).py`             | Runs model training or prediction. Automatically manages input/output.|
| `housing.csv`                   | Original training dataset.                                             |
| `input.csv`                     | Test input generated from test split.                                  |
| `output.csv`                    | Predictions generated from input.csv.                                  |

---

## 🔍 Features

- Full machine learning pipeline using Scikit-learn
- Handles missing values using SimpleImputer
- Encodes categorical column `ocean_proximity` using OneHotEncoder
- Feature scaling with StandardScaler
- Uses ColumnTransformer and Pipeline
- Saves trained model and pipeline as `.pkl` files
- Stores input and prediction results as CSV files

---

## ✅ Requirements

Install required Python packages:


pip install pandas numpy scikit-learn joblib


---


▶️ How to Run the Project

1. Make sure housing.csv is present in the project folder.


2. Open terminal or command prompt in the project directory.


3. Run the main script:



python ML(For - User).py

What Happens When You Run the Script?

If model.pkl and pipeline.pkl do NOT exist:

Splits the dataset

Preprocesses training and test data

Trains the model

Saves model and pipeline

Exports test data to input.csv


If model.pkl and pipeline.pkl already exist:

Loads input.csv

Transforms test data using saved pipeline

Makes predictions

Saves results in output.csv




----


📦 Files Not Included in GitHub

Files like model.pkl and pipeline.pkl may not be included due to size limits. You can regenerate them by simply running the script.


---


📤 Output


After prediction:

output.csv contains:

All input columns from input.csv

A new column median_house_value with predicted house prices




---


👨‍💻 Author

Debashish Parida
GitHub: https://github.com/debashish-5

---
