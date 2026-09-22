import os
import pandas as pd
import numpy as np

DATA_FILE = "data.csv"

def load_dataset(file_path=DATA_FILE):
    """
    Loads and cleans dataset from CSV file.
    Returns cleaned DataFrame.
    """
    if not os.path.exists(file_path):
        return pd.DataFrame(columns=["StudentID", "AvgInternal", "SGPA"])
    
    df = pd.read_csv(file_path)
    # Ensure necessary columns exist and remove invalid rows
    if "AvgInternal" in df.columns:
        df["AvgInternal"] = pd.to_numeric(df["AvgInternal"], errors="coerce")
    if "SGPA" in df.columns:
        df["SGPA"] = pd.to_numeric(df["SGPA"], errors="coerce")
        
    df = df.dropna(subset=["AvgInternal"]).copy()
    return df

def get_student_by_id(df, student_id):
    """
    Searches for a student by StudentID (case-insensitive search).
    """
    if df.empty or "StudentID" not in df.columns or not student_id:
        return pd.DataFrame()
        
    pattern = str(student_id).strip().lower()
    matches = df[df["StudentID"].astype(str).str.lower().str.contains(pattern, na=False)]
    return matches

def batch_predict_df(df, model, scaler):
    """
    Takes a DataFrame containing 'AvgInternal' (marks out of 40),
    runs model prediction in-memory, and returns updated DataFrame.
    """
    if df.empty or "AvgInternal" not in df.columns:
        return df
        
    res_df = df.copy()
    
    # Scale feature
    X = res_df[["AvgInternal"]]
    X_scaled = scaler.transform(X)
    
    # Predict & clip between 0.0 and 10.0
    raw_preds = model.predict(X_scaled)
    res_df["Predicted_SGPA"] = [round(max(0.0, min(10.0, float(p))), 2) for p in raw_preds]
    
    if "SGPA" in res_df.columns:
        res_df["Error"] = (res_df["Predicted_SGPA"] - res_df["SGPA"]).round(2)
        res_df["Abs_Error"] = res_df["Error"].abs().round(2)
        
    return res_df

def process_uploaded_csv(file_buffer, model, scaler):
    """
    Processes an uploaded CSV file completely in-memory without persistent database storage.
    Validates CSV schema and appends predicted SGPA.
    """
    try:
        df = pd.read_csv(file_buffer)
    except Exception as e:
        raise ValueError(f"Could not read CSV file: {e}")
        
    if "AvgInternal" not in df.columns:
        # Check case insensitive match
        col_map = {col.lower(): col for col in df.columns}
        if "avginternal" in col_map:
            df.rename(columns={col_map["avginternal"]: "AvgInternal"}, inplace=True)
        elif "internal" in col_map:
            df.rename(columns={col_map["internal"]: "AvgInternal"}, inplace=True)
        elif "marks" in col_map:
            df.rename(columns={col_map["marks"]: "AvgInternal"}, inplace=True)
        else:
            raise ValueError("CSV file must contain an 'AvgInternal' column (Internal marks out of 40).")
            
    df["AvgInternal"] = pd.to_numeric(df["AvgInternal"], errors="coerce")
    df = df.dropna(subset=["AvgInternal"]).copy()
    
    if df.empty:
        raise ValueError("No valid rows with numerical 'AvgInternal' marks found in uploaded CSV.")
        
    return batch_predict_df(df, model, scaler)

def get_dataset_summary(df):
    """
    Returns high-level summary statistics dictionary for the dataset.
    """
    if df.empty:
        return {}
        
    summary = {
        "total_students": len(df),
        "avg_internal_mean": round(df["AvgInternal"].mean(), 2),
        "avg_internal_min": round(df["AvgInternal"].min(), 2),
        "avg_internal_max": round(df["AvgInternal"].max(), 2),
    }
    
    if "SGPA" in df.columns and not df["SGPA"].isna().all():
        summary.update({
            "sgpa_mean": round(df["SGPA"].mean(), 2),
            "sgpa_min": round(df["SGPA"].min(), 2),
            "sgpa_max": round(df["SGPA"].max(), 2),
        })
        
    return summary
