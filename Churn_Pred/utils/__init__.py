import pandas as pd
import yaml
import os,sys
import numpy as np
import dill, pickle
from pathlib import Path
from Churn_Pred.exception.exception import CustomException


# lowering of column names
def lowercase_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [col.lower() for col in df.columns]
    return df

# `READ YAML` file
def read_yaml_file(filepath:Path) -> dict:
    try:
        with open(filepath, 'rb') as file:
            return yaml.safe_load(file)
    except Exception as e:
        raise CustomException(e,sys)
    
def write_yaml_file(filepath: Path, content: object, replace: bool = False) -> None:
    try:
        # Remove file if `replace=True` and file exists
        if replace and filepath.exists():
            filepath.unlink()

        # Ensure parent directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Write YAML content
        with open(filepath, 'w', encoding='utf-8') as file:
            yaml.dump(content, file, default_flow_style=False)
            
    except Exception as e:
        raise CustomException(e,sys)
    
    # def write_yaml_file(filepath: Path, content: object) -> None:
    #     try:
    #         filepath.parent.mkdir(parents=True, exist_ok=True)
    #         with open(filepath, 'w') as file:
    #             yaml.dump(content, file)
    #     except Exception as e:
    #         raise CustomException(e, sys)


def load_object(filepath: str) -> object:
    try:
        if not os.path.exists(filepath):
            raise Exception(f"File Not Found: {filepath}")
        with open(filepath, 'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys)
    
def load_numpy_array(filepath:str) -> np.array:
    try:
        with open(filepath, 'rb') as f:
            return np.array(f)
    except Exception as e:
        raise CustomException(e,sys)

# def print_overfit_warning(model_name, train_acc, test_acc, threshold):
#     acc_diff = abs(train_acc - test_acc)
#     if acc_diff < threshold:
#         label, color = "Minimal", "\033[92m"
#     elif 0.05 <= acc_diff < 0.10:
#         label, color = "Average", "\033[93m"
#     elif 0.10 <= acc_diff < 0.15:
#         label, color = "High", "\033[91m"
#     else:
#         label, color = "Extreme", "\033[41m"

#     reset = "\033[0m"
#     print(f"{color}⚠️ Overfit/Underfit Detected for {model_name}: "
#           f"Train Acc = {train_acc:.4f}, Test Acc = {test_acc:.4f}, "
#           f"Diff = {acc_diff:.4f} ({label}){reset}")
    
# def print_lift_status(model_name, top_decile_lift, bar=0.5):
#     try:
#         lift_ratio = top_decile_lift / bar

#         if lift_ratio < 0.95:
#             label, rgb = "Below BAR", (139, 0, 0)          # Dark Red
#         elif 0.95 <= lift_ratio <= 1.05:
#             label, rgb = "Around BAR", (255, 215, 0)       # Yellow
#         else:
#             label, rgb = "Above BAR", (50, 205, 50)        # Green

#         r, g, b = rgb
#         color = f"\033[38;2;{r};{g};{b}m"
#         reset = "\033[0m"

#         print(f"{color}📊 Lift Evaluation for {model_name}: "
#               f"Top-Decile Lift = {top_decile_lift:.2f} ({label}){reset}")
        
#     except Exception as e:
#         print(f"❌ Error in lift status display: {e}")



def get_overfit_warning(model_name, train_acc, test_acc, threshold):
    acc_diff = abs(train_acc - test_acc)
    if acc_diff < threshold:
        label = "Minimal"
    elif 0.05 <= acc_diff < 0.10:
        label = "Average"
    elif 0.10 <= acc_diff < 0.15:
        label = "High"
    else:
        label = "Extreme"

    return f"⚠️ Overfit/Underfit Detected for {model_name}: Train Acc = {train_acc:.4f}, Test Acc = {test_acc:.4f}, Diff = {acc_diff:.4f} ({label})"

def get_lift_status(model_name, top_decile_lift, bar=0.5):
    try:
        lift_ratio = top_decile_lift / bar

        if lift_ratio < 0.95:
            label = "Below BAR"
        elif 0.95 <= lift_ratio <= 1.05:
            label = "Around BAR"
        else:
            label = "Above BAR"

        return f"📊 Lift Evaluation for {model_name}: Top-Decile Lift = {top_decile_lift:.2f} ({label})"
        
    except Exception as e:
        return f"❌ Error in lift status display: {e}"

from datetime import datetime
import json
        
def convert_numpy(obj):
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    elif isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()
    else:
        return str(obj)  # fallback


