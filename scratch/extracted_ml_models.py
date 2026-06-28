"import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.neural_network import MLPRegressor
import joblib

# Optional heavy ML imports with safe fallback handling
HAS_XGB = False
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    pass

HAS_TF = False
TF_CLASSES = ()
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM, GRU
    HAS_TF = True
    TF_CLASSES = (Sequential,)
except ImportError:
    pass

HAS_PROPHET = False
try:
    from prophet import Prophet
    HAS_PROPHET = True
except ImportError:
    pass

HAS_STATSMODELS = False
try:
    from statsmodels.tsa.arima.model import ARIMA
    HAS_STATSMODELS = True
except ImportError:
    pass

DATA_PATH = "groundwater_cleaned.csv"


class WeightedEnsembleRegressor:
    """
    A custom ensemble regressor that combines predictions from multiple models
    using pre-calculated weights based on their validation performance.
    """
    def __init__(self, models_with_weights):
        # List of tuples: (model_instance, weight)
        self.models_with_weights = models_with_weights

    def predict(self, X):
        predictions = np.zeros(len(X))
        for model, weight in self.models_with_weights:
            predictions += weight * model.predict(X)
        return predictions


def load_and_preprocess_data(district, station=None):
    """
    Loads groundwater and meteorological data from the CSV database,
    filters by district/station, cleans outliers, and sorts chronologically.
    """
    if not os.path.exists(DATA_PATH):
        return pd.DataFrame()
        
    df = 
<truncated 20894 bytes>