import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DEFAULT_MODEL_FILE = "sgpa_linear_model.pkl"
ALT_MODEL_FILE = "sgpa_linear_model (2).pkl"
DATA_FILE = "data.csv"

def train_linear_model(data_file=DATA_FILE, test_size=0.20, random_state=42, save_path=DEFAULT_MODEL_FILE):
    """
    Trains a Linear Regression model on AvgInternal (out of 40) to predict SGPA.
    Saves production model and scaler to pickle file.
    Returns: (production_model, production_scaler, metrics_dict)
    """
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Dataset file '{data_file}' not found.")
    
    df = pd.read_csv(data_file).dropna(subset=["AvgInternal", "SGPA"])
    
    X = df[["AvgInternal"]]
    y = df["SGPA"]
    
    # Train-test split for evaluation metrics
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    test_scaler = StandardScaler()
    X_train_scaled = test_scaler.fit_transform(X_train)
    X_test_scaled = test_scaler.transform(X_test)
    
    test_model = LinearRegression()
    test_model.fit(X_train_scaled, y_train)
    
    y_pred = test_model.predict(X_test_scaled)
    mae = float(mean_absolute_error(y_test, y_pred))
    mse = float(mean_squared_error(y_test, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_test, y_pred))
    
    metrics = {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "test_count": len(X_test),
        "train_count": len(X_train),
        "total_count": len(df),
        "slope": float(test_model.coef_[0]),
        "intercept": float(test_model.intercept_)
    }
    
    # Train production model on complete dataset
    prod_scaler = StandardScaler()
    X_scaled_full = prod_scaler.fit_transform(X)
    
    prod_model = LinearRegression()
    prod_model.fit(X_scaled_full, y)
    
    # Save model artifact
    save_dict = {"model": prod_model, "scaler": prod_scaler}
    with open(save_path, "wb") as f:
        pickle.dump(save_dict, f)
        
    return prod_model, prod_scaler, metrics

def load_model(file_path=None):
    """
    Loads model and scaler from pickle file.
    Tries candidate paths: user path -> sgpa_linear_model.pkl -> sgpa_linear_model (2).pkl.
    If none found, trains a fresh model automatically.
    Returns: (model, scaler, loaded_file_name)
    """
    candidates = []
    if file_path:
        candidates.append(file_path)
    candidates.extend([DEFAULT_MODEL_FILE, ALT_MODEL_FILE])
    
    for path in candidates:
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    data = pickle.load(f)
                if isinstance(data, dict) and "model" in data and "scaler" in data:
                    return data["model"], data["scaler"], path
            except Exception:
                continue
                
    # Auto-train if model file missing or unreadable
    model, scaler, _ = train_linear_model(save_path=DEFAULT_MODEL_FILE)
    return model, scaler, DEFAULT_MODEL_FILE

def predict_sgpa(internal_mark, model, scaler):
    """
    Predicts SGPA given an internal mark out of 40.
    Returns predicted SGPA clipped between 0.0 and 10.0.
    """
    internal_mark = float(internal_mark)
    internal_mark = max(0.0, min(40.0, internal_mark))
    
    df_input = pd.DataFrame({"AvgInternal": [internal_mark]})
    scaled_input = scaler.transform(df_input)
    raw_pred = model.predict(scaled_input)[0]
    
    sgpa = max(0.0, min(10.0, float(raw_pred)))
    return round(sgpa, 2)
