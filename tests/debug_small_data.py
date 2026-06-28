import os
import sys
import pandas as pd

# Add parent path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from forecast_module.ml_models import load_and_preprocess_data

df_jalna = load_and_preprocess_data("Jalna")
print("Jalna preprocessed shape:", df_jalna.shape)
if not df_jalna.empty:
    print("Jalna columns & data:")
    print(df_jalna[["date", "depth", "temperature", "humidity", "rainfall"]])

df_beed = load_and_preprocess_data("Beed")
print("Beed preprocessed shape:", df_beed.shape)
if not df_beed.empty:
    print("Beed columns & data:")
    print(df_beed[["date", "depth", "temperature", "humidity", "rainfall"]])

df_buldhana = load_and_preprocess_data("Buldhana")
print("Buldhana preprocessed shape:", df_buldhana.shape)
