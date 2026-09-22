import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_FILE = 'data.csv'
MODEL_FILE = 'sgpa_linear_model.pkl'

# Load data
df = pd.read_csv(DATA_FILE)

# Keep only rows where both feature and target are available
df = df.dropna(subset=['AvgInternal', 'SGPA'])

# One feature only: AvgInternal
X = df[['AvgInternal']]
y = df['SGPA']

# Hold out 20% for evaluation
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test) 

# Train simple linear regression
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Evaluate
predictions = model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5
r2 = r2_score(y_test, predictions)

print(f'Test MAE:  {mae:.4f}')
print(f'Test RMSE: {rmse:.4f}')
print(f'Test R²:   {r2:.4f}')
print(f'Slope:     {model.coef_[0]:.6f}')
print(f'Intercept: {model.intercept_:.6f}')

# Retrain with scaled full data for production use
prod_scaler = StandardScaler()
X_production_scaled = prod_scaler.fit_transform(X)

production_model = LinearRegression()
production_model.fit(X_production_scaled, y)

with open(MODEL_FILE, 'wb') as file:
    # Save both objects together in a dictionary
    pickle.dump({'model': production_model, 'scaler': prod_scaler}, file)

print(f'Production model saved to {MODEL_FILE}')

def predict_sgpa_from_percentage(percentage):
    value = pd.DataFrame({'AvgInternal': [percentage]})
    # Scale the input using the production scaler before predicting
    value_scaled = prod_scaler.transform(value)
    return float(production_model.predict(value_scaled)[0])


