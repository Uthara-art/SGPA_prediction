import pickle
import pandas as pd

MODEL_FILE = "sgpa_linear_model.pkl"

# Load the already-trained model and scaler
with open(MODEL_FILE, "rb") as file:
    data = pickle.load(file)

model = data["model"]
scaler = data["scaler"]

# Get your internal mark
internal = float(input("Enter your internal mark out of 40: "))

# Put input into the same format used during training
X = pd.DataFrame({
    "AvgInternal": [internal]
})

# Scale using the ALREADY fitted scaler
X_scaled = scaler.transform(X)

# Predict SGPA using the ALREADY trained model
predicted_sgpa = model.predict(X_scaled)[0]

print(f"\nInternal Mark: {internal:.2f}/40")
print(f"Predicted SGPA: {predicted_sgpa:.2f}")