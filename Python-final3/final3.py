# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing(as_frame=True)
data = housing.frame

print("Dataset Head:")
print(data.head())

data = data.dropna()

X = data[['MedInc', 'HouseAge', 'AveRooms', 'AveOccup']]
y = data['MedHouseVal']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

print(f"Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_}")

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], '--r', linewidth=2)
plt.title('Actual vs Predicted Values')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.show()

def predict_median_house_value(med_inc, house_age, ave_rooms, ave_occup):
    features = np.array([[med_inc, house_age, ave_rooms, ave_occup]])
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)
    return prediction[0]


print("Predicting Median House Value...")
user_input = [3.0, 15, 5.0, 2.0] 
predicted_value = predict_median_house_value(*user_input)
print(f"Predicted Median House Value: {predicted_value}")
print("\nChallenges and Limitations:")
print("- Linear regression assumes a linear relationship between features and the target variable.")
print("- Outliers can significantly affect the model's performance.")
print("- The dataset might have multicollinearity issues.")
print("\nFuture Improvements:")
print("- Explore polynomial regression or other regression techniques.")
print("- Include more features or use feature engineering techniques.")
print("- Use cross-validation for better model evaluation.")
