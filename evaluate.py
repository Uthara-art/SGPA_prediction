import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Load data
df = pd.read_csv("data.csv")

X = df[["AvgInternal"]]
y = df["SGPA"]

# Same split used by Group 3
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# Load model and scaler
data = joblib.load("sgpa_linear_model.pkl")
model = data["model"]
scaler = data["scaler"]

# Scale test data
X_test_scaled = scaler.transform(X_test)

# Predictions
predictions = model.predict(X_test_scaled)

# Evaluation metrics
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

# Results
print("\n========== GROUP 4 MODEL EVALUATION ==========")
print(f"Test samples : {len(y_test)}")
print(f"MAE          : {mae:.4f}")
print(f"MSE          : {mse:.4f}")
print(f"RMSE         : {rmse:.4f}")
print(f"R²           : {r2:.4f}")
print("==============================================")

# Actual vs Predicted table
results = pd.DataFrame({
    "Actual_SGPA": y_test.values,
    "Predicted_SGPA": predictions,
    "Residual": y_test.values - predictions
})

print("\nActual vs Predicted:")
print(results.to_string(index=False))

# Actual vs Predicted graph
plt.figure()
plt.scatter(y_test, predictions)
plt.xlabel("Actual SGPA")
plt.ylabel("Predicted SGPA")
plt.title("Actual vs Predicted SGPA")
plt.savefig("actual_vs_predicted.png", dpi=300, bbox_inches="tight")
plt.show()

# Residual plot
residuals = y_test - predictions

plt.figure()
plt.scatter(predictions, residuals)
plt.axhline(y=0, linestyle="--")
plt.xlabel("Predicted SGPA")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.savefig("residual_plot.png", dpi=300, bbox_inches="tight")
plt.show()