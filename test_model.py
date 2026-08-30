import pandas as pd
import pickle

MODEL_FILE = "sgpa_linear_model.pkl"

# Load model and scaler
with open(MODEL_FILE, "rb") as file:
    saved_data = pickle.load(file)

model = saved_data["model"]
scaler = saved_data["scaler"]


def predict_from_percentage(percentage):
    # Convert percentage into DataFrame
    input_data = pd.DataFrame({
        "AvgInternal": [percentage]
    })

    # Apply the SAME scaling used during training
    input_scaled = scaler.transform(input_data)

    # Predict SGPA
    prediction = model.predict(input_scaled)

    return float(prediction[0])


def calculate_percentage(marks, full_marks):
    total = 0

    for mark, full_mark in zip(marks, full_marks):
        total += mark / full_mark

    return (total / len(marks)) * 100


def predict_from_marks(marks, full_marks):
    percentage = calculate_percentage(marks, full_marks)
    sgpa = predict_from_percentage(percentage)

    return percentage, sgpa


# --------------------------------
# TEST 1: Percentage directly
# --------------------------------

percentage = 89.5

sgpa = predict_from_percentage(percentage)

print("----- Percentage Test -----")
print("Input percentage:", percentage)
print("Predicted SGPA:", round(sgpa, 2))


# --------------------------------
# TEST 2: Individual subject marks
# --------------------------------

marks = [80, 75, 42, 90]
full_marks = [100, 100, 50, 100]

percentage, sgpa = predict_from_marks(marks, full_marks)

print("\n----- Subject Marks Test -----")
print("Marks:", marks)
print("Full marks:", full_marks)
print("Calculated percentage:", round(percentage, 2))
print("Predicted SGPA:", round(sgpa, 2))