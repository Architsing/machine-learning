# Gender Prediction model

## Overview

This project explores logistic regression for predicting the gender label in a small survey dataset from four categorical preference features. It includes exploratory data analysis (EDA) and a Streamlit app for trying predictions.

This is an educational demonstration. Its results are not reliable enough to infer an individual’s gender.

## Features

- EDA charts showing gender counts across each preference category
- Logistic regression with one-hot encoded categorical features
- A Streamlit form for entering preferences and viewing a prediction
- Saved model and feature metadata loaded with Joblib

## Model inputs

The model uses:

- Favorite Color
- Favorite Music Genre
- Favorite Beverage
- Favorite Soft Drink

The target is `Gender`, encoded as `F = 0` and `M = 1`.

## Evaluation

The dataset contains 66 rows, with 33 examples in each gender class.

| Evaluation | Result |
|---|---:|
| Test accuracy | 71.4% (10 of 14 test rows) |
| Mean 5-fold cross-validation accuracy | 56.3% |
| Cross-validation score range | 38.5%–76.9% |

The cross-validation scores vary considerably between folds. With so few rows, the results are sensitive to how the data is split and should be treated as preliminary.

## Run locally

Install the required packages:

```bash
pip install -r requirements.txt
```

Start the app:

```bash
streamlit run app.py
```

Keep the dataset CSV and the three Joblib files—`gender_model.pkl`, `feature_columns.pkl`, and `feature_categories.pkl`—in the same folder as `app.py`.

## Streamlit Community Cloud

Deploy the app from the `https://machine-learning-npkcbc3gqswezdscgatyzy.streamlit.app/` GitHub repository using this entrypoint:

```text
gender-predictor-model/app.py
```

Ensure `requirements.txt`, the dataset CSV, and the three model files are committed in the `gender-predictor-model` folder.
