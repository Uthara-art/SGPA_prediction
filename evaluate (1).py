import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("data.csv")

X = df[["AvgInternal"]]
y = df["SGPA"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled, y_train)

predictions = model.predict(X_test_scaled)

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = mse ** 0.5
r2 = r2_score(y_test, predictions)

print("===== GROUP 4 EVALUATION =====")
print("Total students :", len(df))
print("Training       :", len(X_train))
print("Testing        :", len(X_test))
print(f"MAE            : {mae:.4f}")
print(f"MSE            : {mse:.4f}")
print(f"RMSE           : {rmse:.4f}")
print(f"R²             : {r2:.4f}")

# Actual vs Predicted
plt.figure()
plt.scatter(y_test, predictions)
plt.xlabel("Actual SGPA")
plt.ylabel("Predicted SGPA")
plt.title("Actual vs Predicted SGPA")
plt.savefig("actual_vs_predicted.png", dpi=300, bbox_inches="tight")
plt.show()

# Residual Plot
residuals = y_test - predictions

plt.figure()
plt.scatter(predictions, residuals)
plt.axhline(0, linestyle="--")
plt.xlabel("Predicted SGPA")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.savefig("residual_plot.png", dpi=300, bbox_inches="tight")
plt.show()


results = pd.DataFrame({
    "Actual_SGPA": y_test.values,
    "Predicted_SGPA": predictions,
    "Residual": y_test.values - predictions
})

results.to_csv("test_results.csv", index=False)
print("\nSaved: test_results.csv")