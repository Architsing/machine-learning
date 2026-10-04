humidity-clf

A decision tree classifier that predicts whether humidity will be high at 3 p.m. using the weather readings taken at 9 a.m., with a small Streamlit web app on top.

Problem

Given nine weather measurements taken at 9 a.m., predict a binary label:

1 = high humidity at 3 p.m.
0 = humidity not high at 3 p.m.

This is a supervised classification problem.

Dataset

daily_weather.csv contains daily weather readings (1,095 days).

Type	Columns
Features (9)	air_pressure_9am, 
air_temp_9am
avg_wind_direction_9am
avg_wind_speed_9am
max_wind_direction_9am
max_wind_speed_9am
rain_accumulation_9am
rain_duration_9am
relative_humidity_9am
Target	high_humidity_3pm

The target is balanced (548 days of class 0, 547 days of class 1), so accuracy is a fair metric.

Approach
EDA: head, describe, histograms, value_counts, and a correlation heatmap.
Missing values: 31 rows (about 3%) had missing values and were dropped, leaving 1,064 rows.
Split: 80% train (851 rows) and 20% test (213 rows), random_state=42.
Model: DecisionTreeClassifier. No feature scaling was needed, since decision trees are not affected by feature scale.
Overfitting control: limited the tree depth with max_depth=3.
Evaluation: accuracy and confusion matrix on both train and test data.
EDA findings
relative_humidity_9am has the strongest correlation with the target (0.68).
air_pressure_9am is the next strongest (-0.49).
avg_wind_speed_9am and max_wind_speed_9am are almost perfectly correlated (1.00), so they carry nearly the same information.
rain_duration_9am has extreme outliers (most days are 0, the maximum is 17,704).


Results

Model	Train accuracy	Test accuracy	Gap

Decision tree, no depth limit	100%	86.4%	13.6%

Decision tree, max_depth=3	89.7%	88.3%	1.4%

The unrestricted tree memorized the training data (100% train accuracy) and did worse on unseen days. Limiting the depth lowered train accuracy slightly, raised test accuracy, and almost closed the gap.

Confusion matrix on the test set (max_depth=3):

	Predicted not high	Predicted high
Actual not high	91	11
Actual high	14	97
Streamlit app

The app has three tabs:

Predict: enter the 9 a.m. readings and get a prediction with the model's confidence.
Model results: accuracy, confusion matrix, feature importance, and the tree itself.
Data exploration: the EDA from the notebook.
Run it locally
bash
pip install -r requirements.txt
python -m streamlit run app.py

app.py, humidity_tree.joblib and daily_weather.csv must be in the same folder. If loading the model fails with a version error, retrain it in the notebook using the same scikit-learn version as in requirements.txt and save it again.

Limitations
The test set has only 213 rows, so a difference of one or two percentage points between models can be noise. Results were not cross-validated yet.
Only max_depth=3 has been tried so far, not a full search.
The model only uses 9 a.m. readings from one location, so this is a learning project and not a real weather forecast.
What I learned
Why 100% train accuracy is a warning sign, not a goal.
How max_depth controls overfitting in decision trees.
How to compare train and test accuracy to spot overfitting.
Decision trees do not need feature scaling.
How to save a model with joblib and serve it with Streamlit.
