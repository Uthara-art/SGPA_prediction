import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def compute_eval_metrics(df_results):
    """
    Computes regression evaluation metrics: MAE, MSE, RMSE, and R2.
    Expects DataFrame with 'SGPA' and 'Predicted_SGPA' columns.
    """
    if df_results.empty or "SGPA" not in df_results.columns or "Predicted_SGPA" not in df_results.columns:
        return {}
        
    valid = df_results.dropna(subset=["SGPA", "Predicted_SGPA"])
    if valid.empty:
        return {}
        
    y_true = valid["SGPA"]
    y_pred = valid["Predicted_SGPA"]
    
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))
    
    return {
        "count": len(valid),
        "mae": round(mae, 4),
        "mse": round(mse, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4)
    }

def create_styled_figure(figsize=(7, 4.5)):
    """
    Creates a styled matplotlib figure matching the app's dark theme palette.
    """
    fig, ax = plt.subplots(figsize=figsize, facecolor="#0f172a")
    ax.set_facecolor("#1e293b")
    
    # Custom colors
    ax.tick_params(colors="#94a3b8", labelsize=10)
    for spine in ax.spines.values():
        spine.set_color("#334155")
        spine.set_linewidth(1.2)
        
    ax.grid(True, linestyle="--", alpha=0.2, color="#94a3b8")
    return fig, ax

def plot_actual_vs_predicted(df_results):
    """
    Generates Actual vs Predicted SGPA scatter plot.
    """
    fig, ax = create_styled_figure()
    
    if "SGPA" in df_results.columns and "Predicted_SGPA" in df_results.columns:
        y_true = df_results["SGPA"]
        y_pred = df_results["Predicted_SGPA"]
        
        # Scatter points
        ax.scatter(y_true, y_pred, color="#6366f1", alpha=0.85, edgecolors="#a5b4fc", s=60, label="Students")
        
        # Identity reference line (y = x)
        min_val = min(y_true.min(), y_pred.min()) - 0.5
        max_val = max(y_true.max(), y_pred.max()) + 0.5
        ax.plot([min_val, max_val], [min_val, max_val], color="#ef4444", linestyle="--", linewidth=1.8, label="Ideal (y = x)")
        
        ax.set_xlabel("Actual SGPA", color="#cbd5e1", fontsize=11, fontweight="bold")
        ax.set_ylabel("Predicted SGPA", color="#cbd5e1", fontsize=11, fontweight="bold")
        ax.set_title("Actual vs Predicted SGPA", color="#f8fafc", fontsize=13, fontweight="bold", pad=12)
        ax.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#cbd5e1")
        
    plt.tight_layout()
    return fig

def plot_residuals(df_results):
    """
    Generates Residuals vs Predicted SGPA scatter plot.
    """
    fig, ax = create_styled_figure()
    
    if "SGPA" in df_results.columns and "Predicted_SGPA" in df_results.columns:
        y_true = df_results["SGPA"]
        y_pred = df_results["Predicted_SGPA"]
        residuals = y_true - y_pred
        
        ax.scatter(y_pred, residuals, color="#38bdf8", alpha=0.85, edgecolors="#7dd3fc", s=60)
        ax.axhline(0, color="#f59e0b", linestyle="--", linewidth=1.8, label="Zero Error Line")
        
        ax.set_xlabel("Predicted SGPA", color="#cbd5e1", fontsize=11, fontweight="bold")
        ax.set_ylabel("Residual (Actual - Predicted)", color="#cbd5e1", fontsize=11, fontweight="bold")
        ax.set_title("Residual Plot (Model Errors)", color="#f8fafc", fontsize=13, fontweight="bold", pad=12)
        ax.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#cbd5e1")
        
    plt.tight_layout()
    return fig

def plot_error_distribution(df_results):
    """
    Generates Histogram of Absolute Prediction Errors.
    """
    fig, ax = create_styled_figure()
    
    if "Abs_Error" in df_results.columns:
        errors = df_results["Abs_Error"]
        ax.hist(errors, bins=12, color="#10b981", alpha=0.75, edgecolor="#34d399", rwidth=0.85)
        ax.set_xlabel("Absolute Error in SGPA", color="#cbd5e1", fontsize=11, fontweight="bold")
        ax.set_ylabel("Number of Students", color="#cbd5e1", fontsize=11, fontweight="bold")
        ax.set_title("Absolute Error Distribution", color="#f8fafc", fontsize=13, fontweight="bold", pad=12)
        
    plt.tight_layout()
    return fig
