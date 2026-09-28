# 🌍 Life Expectancy Prediction

This project predicts a country's **Life Expectancy** from health, economic and social indicators using **Multiple Linear Regression**, and serves the model as a live **Streamlit** web app.

## 🚀 Live Demo

👉 [Life Expectancy Predictor](https://machine-learning-wzdycri6rviekaqlusqw84.streamlit.app/)

Select a country and status, enter the indicator values, and get the predicted life expectancy.

## 📊 Dataset

Country-level data (2938 rows, 22 columns) covering health, economic and demographic indicators such as Adult Mortality, GDP, Schooling, BMI, HIV/AIDS, Alcohol and Income composition of resources. The target is **Life expectancy**.

## 🔍 Data Exploration

* Checked the dataset shape, data types and summary statistics
* Identified missing values in each column
* Checked category counts for the `Status` column

## 🧹 Data Preparation

* **Missing values:** filled with the **median** of each column (median is less affected by outliers than the mean)
* **Status (Developing/Developed):** encoded into one binary column, `Developing_Status`. Only one column is kept to avoid the **dummy variable trap**
* **Country (190+ unique values):** encoded with **LabelEncoder**, because One-Hot Encoding would create 190+ extra columns
* Split the data into **80% train / 20% test**

## 🤖 Machine Learning

* Model: **Linear Regression** (scikit-learn), trained on 21 features
* Evaluated on both train and test data using **MSE, MAE and MAPE**
* Inspected the model coefficients (`coef_`) to see which features influence the prediction most

## 📈 Results

| Metric | Train | Test |
|---|---|---|
| MAE | 3.04 | 2.85 |
| MAPE | 4.65% | 4.38% |
| MSE | 16.55 | 15.14 |

Train and test errors are very close, so the model is **not overfitting**.

## 🌐 Model Deployment

The trained model, the country encoder and the feature list were saved with **joblib**, then used in a Streamlit app (`app.py`) deployed on **Streamlit Community Cloud** through GitHub.

## 🛠️ Technologies Used

* Python
* Pandas, NumPy
* Scikit-learn
* Joblib
* Streamlit
* Google Colab
* GitHub

## 📁 Files

* `life_expectancy_regression.ipynb` – Colab notebook (data preparation, training, evaluation)
* `CV Data Life Expectancy Data (1).csv` – dataset
* `app.py` – Streamlit web application
* `linear_regression_model.pkl` – trained Linear Regression model
* `country_encoder.pkl` – fitted Country LabelEncoder
* `feature_columns.pkl` – feature names in the order the model expects
* `requirements.txt` – Python dependencies

## 🔗 Project Links

* **Live Demo:** https://machine-learning-wzdycri6rviekaqlusqw84.streamlit.app/
* **Source Code:** https://github.com/Architsing/machine-learning/tree/main/life-expectancy-project
