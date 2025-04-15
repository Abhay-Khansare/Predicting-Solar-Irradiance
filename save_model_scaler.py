# This script trains the model and saves the scaler and model as .pkl files

import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("SolarPrediction.csv")

# Extract date features
df['Data'] = df['Data'].apply(lambda x: x.split()[0])
df['Month'] = pd.to_datetime(df['Data']).dt.month
df['Day'] = pd.to_datetime(df['Data']).dt.day
df['Hour'] = pd.to_datetime(df['Time']).dt.hour
df['Minute'] = pd.to_datetime(df['Time']).dt.minute
df['Second'] = pd.to_datetime(df['Time']).dt.second

# Extract sunrise/sunset time features
df['risehour'] = df['TimeSunRise'].apply(lambda x: int(re.search(r'^\d+', x).group(0)))
df['riseminuter'] = df['TimeSunRise'].apply(lambda x: int(re.search(r'(?<=:)[0-9]+(?=:)', x).group(0)))
df['sethour'] = df['TimeSunSet'].apply(lambda x: int(re.search(r'^\d+', x).group(0)))
df['setminute'] = df['TimeSunSet'].apply(lambda x: int(re.search(r'(?<=:)[0-9]+(?=:)', x).group(0)))

# Drop unnecessary columns
df.drop(['UNIXTime', 'Data', 'Time', 'TimeSunRise', 'TimeSunSet'], axis=1, inplace=True)

# Define features and target
X = df.drop('Radiation', axis=1)
y = df['Radiation']

# Scale features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Model Performance:")
print(f" - RMSE: {rmse:.2f}")
print(f" - R^2 Score: {r2:.4f}")

# Visualize Actual vs Predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='dodgerblue', alpha=0.5, label='Predictions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Radiation')
plt.ylabel('Predicted Radiation')
plt.title('Actual vs Predicted Solar Radiation')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save plot to file
plt.savefig("prediction_vs_actual.png")
print("📊 Prediction vs Actual plot saved as prediction_vs_actual.png")

# Save model and scaler
joblib.dump(model, 'xgb_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Model and scaler saved successfully.")