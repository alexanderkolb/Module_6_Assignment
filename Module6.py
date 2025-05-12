import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt


#Read in and clean dataset
df = pd.read_csv('Students_Grading_Dataset.csv', low_memory = False)
df = df.dropna()
df = df.drop_duplicates()

#Determine features and set target
values = df[['Attendance (%)', 'Midterm_Score', 'Assignments_Avg', 'Quizzes_Avg', 'Study_Hours_per_Week', 'Stress_Level (1-10)', 'Sleep_Hours_per_Night']]
target = df['Final_Score']

x_train, x_test, y_train, y_test = train_test_split(values, target, test_size=0.2, random_state=10)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(x_train, y_train)

#make predictions on the training and test data
y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)

#evaluate model
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f'Train MAE: {train_mae:,.2f}')
print(f'Test MAE: {test_mae:,.2f}')
print(f'Train R²: {train_r2:.2f}')
print(f'Test R²: {test_r2:.2f}')

#find top 5 worst predictions
errors = np.abs(y_test_pred - y_test.to_numpy())
top_errors = errors.argsort()[-5:] 

for idx in top_errors:
    print(f"Predicted: {y_test_pred[idx]:,.0f} | Actual: {y_test.iloc[idx]:,.0f}")

#plot predicted vs actual
plt.scatter(y_test, y_test_pred)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.xlabel('Actual Test Score')
plt.ylabel('Predicted Test Score')
plt.title('Predicted vs Actual Test Score')
plt.show()

