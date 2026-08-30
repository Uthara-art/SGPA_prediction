import pandas as pd
import pickle

MODEL_FILE = "sgpa_linear_model.pkl"
DATA_FILE = "data.csv"

# Load trained model and scaler
with open(MODEL_FILE, "rb") as file:
    saved_data = pickle.load(file)

model = saved_data["model"]
scaler = saved_data["scaler"]

# Load dataset
df = pd.read_csv(DATA_FILE)

# Remove rows where X or y is missing
df = df.dropna(subset=["AvgInternal", "SGPA"])

# Get the feature
X = df[["AvgInternal"]]

# Scale the feature using the SAME scaler used during training
X_scaled = scaler.transform(X)

# Predict SGPA
df["Predicted_SGPA"] = model.predict(X_scaled)

# Calculate prediction error
df["Error"] = df["Predicted_SGPA"] - df["SGPA"]
df["Absolute_Error"] = df["Error"].abs()

# Display results
result = df[
    ["StudentID", "AvgInternal", "SGPA", "Predicted_SGPA", "Error"]
]

print(result.round(2).to_string(index=False))

# Overall performance
mae = df["Absolute_Error"].mean()
rmse = (df["Error"] ** 2).mean() ** 0.5

print("\n-----------------------------")
print("MAE :", round(mae, 4))
print("RMSE:", round(rmse, 4))