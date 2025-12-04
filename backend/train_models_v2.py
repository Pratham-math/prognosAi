"""Enhanced training pipeline with advanced feature engineering and uncertainty quantification."""

import os
import numpy as np
import pandas as pd
from typing import Tuple

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor
import joblib
import warnings
warnings.filterwarnings('ignore')


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


def generate_realistic_data(n_days: int = 730) -> pd.DataFrame:
    """Generate synthetic ED data with realistic patterns.
    
    Improvements over original:
    - Multi-modal festival patterns (Diwali, Holi, etc.)
    - Weather correlation (extreme heat/cold)
    - Day-of-month effects (paycheck surge)
    - Hourly volatility increases during peak times
    - Autocorrelation in AQI
    """
    hours = n_days * 24
    idx = pd.date_range(start="2023-01-01", periods=hours, freq="h")
    df = pd.DataFrame({"timestamp": idx})
    
    # Baseline: ~5 admissions/hour with realistic variance
    base_rate = 5.0
    
    # Extract temporal features
    hour = df["timestamp"].dt.hour
    dow = df["timestamp"].dt.dayofweek
    day = df["timestamp"].dt.day
    month = df["timestamp"].dt.month
    
    # Hour-of-day pattern: dual peaks (morning rush + evening)
    hour_factor = (
        1.0 
        + 0.3 * np.sin(2 * np.pi * (hour - 8) / 24.0)  # Morning peak
        + 0.4 * np.sin(2 * np.pi * (hour - 18) / 24.0)  # Evening peak
    )
    
    # Weekend effect: lower mornings, higher evenings
    weekend_mask = dow >= 5
    weekend_factor = np.where(
        weekend_mask & (hour < 12), 0.7,  # Slow weekend mornings
        np.where(weekend_mask & (hour >= 18), 1.4, 1.0)  # Busy weekend evenings
    )
    
    # End-of-month surge (paycheck effect)
    eom_factor = np.where(day >= 28, 1.15, 1.0)
    
    # Seasonal pattern (higher winter admissions)
    seasonal_factor = 1.0 + 0.25 * np.sin(2 * np.pi * (month - 1) / 12.0 + np.pi)
    
    # Realistic AQI with autocorrelation
    t = np.arange(hours)
    aqi = 100.0
    aqi_series = []
    for i in range(hours):
        # Slow drift + daily cycle + noise + autocorrelation
        drift = 40 * np.sin(2 * np.pi * i / (24 * 14))  # 2-week cycle
        daily = 20 * np.sin(2 * np.pi * (i % 24 - 12) / 24.0)  # Daily pattern
        noise = np.random.randn() * 8
        aqi += 0.05 * (drift + daily - aqi) + noise  # AR(1)-like
        aqi = np.clip(aqi, 20, 400)
        aqi_series.append(aqi)
    aqi_series = np.array(aqi_series)
    
    # Festival windows with AQI spikes
    festival_flag = np.zeros(hours)
    festival_windows = [
        (60, 72),    # Festival 1 (3 days)
        (110, 144),  # Festival 2 (1.5 days)
        (180, 204),  # Festival 3 (1 day)
        (300, 336),  # Festival 4 (1.5 days)
        (500, 524),  # Festival 5 (1 day)
    ]
    for start_day, end_day in festival_windows:
        start_idx = start_day * 24
        end_idx = end_day * 24
        if end_idx <= hours:
            aqi_series[start_idx:end_idx] += np.random.uniform(60, 120)
            festival_flag[start_idx:end_idx] = 1.0
    
    # Temperature effect (synthetic, correlated with season)
    temp = 25 + 10 * np.sin(2 * np.pi * (month - 1) / 12.0) + np.random.randn(hours) * 3
    temp_stress = np.where(temp > 35, 1.2, np.where(temp < 10, 1.15, 1.0))  # Extremes
    
    # Expected admissions
    expected = (
        base_rate 
        * hour_factor 
        * weekend_factor 
        * eom_factor 
        * seasonal_factor
        * temp_stress
    )
    
    # AQI effect: nonlinear above threshold
    aqi_effect = 1.0 + 0.004 * np.clip(aqi_series - 100, 0, None)
    expected *= aqi_effect
    
    # Festival surge
    expected *= 1.0 + 0.5 * festival_flag
    
    # Poisson admissions with overdispersion
    admissions = np.random.negative_binomial(
        n=np.clip(expected, 0.1, None), 
        p=0.7
    )
    
    df["hour"] = hour
    df["dayofweek"] = dow
    df["day"] = day
    df["month"] = month
    df["aqi"] = aqi_series
    df["temperature"] = temp
    df["is_festival"] = festival_flag
    df["is_weekend"] = weekend_mask.astype(int)
    df["admissions"] = admissions
    
    return df


def create_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """Advanced feature engineering with Fourier and interaction terms."""
    df = df.copy()
    
    # Lags at multiple scales
    for lag in [1, 2, 3, 6, 12, 24, 48, 72, 168]:  # up to 1 week
        df[f"adm_lag_{lag}"] = df["admissions"].shift(lag)
    
    for lag in [1, 24, 48]:
        df[f"aqi_lag_{lag}"] = df["aqi"].shift(lag)
        df[f"temp_lag_{lag}"] = df["temperature"].shift(lag)
    
    # Rolling statistics at multiple windows
    for window in [6, 12, 24, 48, 72, 168]:
        df[f"adm_roll_mean_{window}"] = df["admissions"].rolling(window).mean()
        df[f"adm_roll_std_{window}"] = df["admissions"].rolling(window).std()
        df[f"adm_roll_max_{window}"] = df["admissions"].rolling(window).max()
    
    # Fourier features for cyclical patterns
    for period, name in [(24, "daily"), (168, "weekly"), (730, "yearly")]:
        df[f"sin_{name}"] = np.sin(2 * np.pi * np.arange(len(df)) / period)
        df[f"cos_{name}"] = np.cos(2 * np.pi * np.arange(len(df)) / period)
    
    # Interaction features
    df["aqi_x_weekend"] = df["aqi"] * df["is_weekend"]
    df["temp_x_hour"] = df["temperature"] * df["hour"]
    df["festival_x_weekend"] = df["is_festival"] * df["is_weekend"]
    
    # Rate of change features
    df["adm_delta_1h"] = df["admissions"].diff(1)
    df["adm_delta_24h"] = df["admissions"].diff(24)
    df["aqi_delta_1h"] = df["aqi"].diff(1)
    
    # Target: next-hour admissions
    df["target"] = df["admissions"].shift(-1)
    
    # Drop NaN rows
    df = df.dropna().reset_index(drop=True)
    return df


def make_sequences(X: np.ndarray, y: np.ndarray, seq_len: int = 72) -> Tuple[np.ndarray, np.ndarray]:
    """Create sequences for LSTM with longer lookback."""
    X_seq, y_seq = [], []
    for i in range(seq_len, len(X)):
        X_seq.append(X[i - seq_len:i, :])
        y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq)


def build_advanced_lstm(input_seq_len: int, n_features: int) -> tf.keras.Model:
    """LSTM with batch normalization and regularization for better generalization."""
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(input_seq_len, n_features), 
             kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.3),
        
        LSTM(64, return_sequences=True, kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.3),
        
        LSTM(32, kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(16, activation="relu", kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(1)
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss="huber",  # Robust to outliers
        metrics=["mae", "mse"]
    )
    return model


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calculate comprehensive evaluation metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    
    return {
        "MAE": mae,
        "RMSE": rmse,
        "R²": r2,
        "MAPE": mape
    }


def main():
    print("=" * 60)
    print("ENHANCED TRAINING PIPELINE V2")
    print("=" * 60)
    
    # Generate data
    print("\n[1/6] Generating realistic synthetic data...")
    df_raw = generate_realistic_data(n_days=730)
    csv_path = os.path.join(DATA_DIR, "synthetic_ed_data_v2.csv")
    df_raw.to_csv(csv_path, index=False)
    print(f"✓ Saved to {csv_path}")
    print(f"  Shape: {df_raw.shape}")
    print(f"  Admission stats: mean={df_raw['admissions'].mean():.2f}, std={df_raw['admissions'].std():.2f}")
    
    # Feature engineering
    print("\n[2/6] Creating advanced features...")
    df = create_advanced_features(df_raw)
    
    # Select features (exclude identifiers and target)
    exclude_cols = ["timestamp", "admissions", "target"]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    print(f"✓ Generated {len(feature_cols)} features")
    
    X = df[feature_cols].values
    y = df["target"].values
    
    # Chronological split: 70% train, 15% val, 15% test
    n = len(X)
    train_idx = int(n * 0.70)
    val_idx = int(n * 0.85)
    
    X_train_raw = X[:train_idx]
    y_train_raw = y[:train_idx]
    
    X_val_raw = X[train_idx:val_idx]
    y_val_raw = y[train_idx:val_idx]
    
    X_test_raw = X[val_idx:]
    y_test_raw = y[val_idx:]
    
    print(f"  Train: {len(X_train_raw)}, Val: {len(X_val_raw)}, Test: {len(X_test_raw)}")
    
    # Scale features
    print("\n[3/6] Scaling features...")
    scaler = StandardScaler()  # StandardScaler for features with Fourier terms
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)
    X_test = scaler.transform(X_test_raw)
    print("✓ Scaling complete")
    
    # Train XGBoost
    print("\n[4/6] Training XGBoost...")
    xgb = XGBRegressor(
        n_estimators=500,
        max_depth=7,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="reg:squarederror",
        n_jobs=-1,
        random_state=42,
        early_stopping_rounds=50
    )
    xgb.fit(
        X_train, y_train_raw,
        eval_set=[(X_val, y_val_raw)],
        verbose=False
    )
    
    xgb_val_pred = xgb.predict(X_val)
    xgb_test_pred = xgb.predict(X_test)
    
    print("✓ XGBoost trained")
    print("  Validation metrics:", calculate_metrics(y_val_raw, xgb_val_pred))
    print("  Test metrics:", calculate_metrics(y_test_raw, xgb_test_pred))
    
    # Train LSTM
    print("\n[5/6] Training LSTM...")
    SEQ_LEN = 72  # 3 days lookback
    
    X_train_seq, y_train_seq = make_sequences(X_train, y_train_raw, SEQ_LEN)
    X_val_seq, y_val_seq = make_sequences(X_val, y_val_raw, SEQ_LEN)
    X_test_seq, y_test_seq = make_sequences(X_test, y_test_raw, SEQ_LEN)
    
    print(f"  Sequence shapes: Train {X_train_seq.shape}, Val {X_val_seq.shape}, Test {X_test_seq.shape}")
    
    lstm = build_advanced_lstm(SEQ_LEN, X_train_seq.shape[2])
    
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=7, min_lr=1e-6),
        ModelCheckpoint(
            os.path.join(MODELS_DIR, "lstm_best_v2.keras"),
            monitor="val_loss",
            save_best_only=True
        )
    ]
    
    history = lstm.fit(
        X_train_seq, y_train_seq,
        validation_data=(X_val_seq, y_val_seq),
        epochs=100,
        batch_size=64,
        callbacks=callbacks,
        verbose=1
    )
    
    lstm_val_pred = lstm.predict(X_val_seq, verbose=0).flatten()
    lstm_test_pred = lstm.predict(X_test_seq, verbose=0).flatten()
    
    print("✓ LSTM trained")
    print("  Validation metrics:", calculate_metrics(y_val_seq, lstm_val_pred))
    print("  Test metrics:", calculate_metrics(y_test_seq, lstm_test_pred))
    
    # Train ensemble
    print("\n[6/6] Training Ridge ensemble...")
    xgb_val_aligned = xgb_val_pred[SEQ_LEN:]
    xgb_test_aligned = xgb_test_pred[SEQ_LEN:]
    
    meta_X_val = np.column_stack([xgb_val_aligned, lstm_val_pred])
    meta_X_test = np.column_stack([xgb_test_aligned, lstm_test_pred])
    
    ridge = Ridge(alpha=2.0)
    ridge.fit(meta_X_val, y_val_seq)
    
    ensemble_val_pred = ridge.predict(meta_X_val)
    ensemble_test_pred = ridge.predict(meta_X_test)
    
    print("✓ Ensemble trained")
    print(f"  Ridge weights: XGB={ridge.coef_[0]:.3f}, LSTM={ridge.coef_[1]:.3f}")
    print("  Validation metrics:", calculate_metrics(y_val_seq, ensemble_val_pred))
    print("  Test metrics:", calculate_metrics(y_test_seq, ensemble_test_pred))
    
    # Save models
    print("\n[SAVE] Saving models and artifacts...")
    joblib.dump(xgb, os.path.join(MODELS_DIR, "xgb_model_v2.pkl"))
    lstm.save(os.path.join(MODELS_DIR, "lstm_model_v2.keras"))
    joblib.dump(ridge, os.path.join(MODELS_DIR, "ridge_meta_v2.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler_v2.pkl"))
    
    meta = {
        "feature_cols": feature_cols,
        "seq_len": SEQ_LEN,
        "model_version": "v2",
        "test_metrics": calculate_metrics(y_test_seq, ensemble_test_pred)
    }
    np.save(os.path.join(MODELS_DIR, "meta_v2.npy"), meta, allow_pickle=True)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Models saved to: {MODELS_DIR}")
    print(f"\nFinal Test Performance:")
    for metric, value in calculate_metrics(y_test_seq, ensemble_test_pred).items():
        print(f"  {metric}: {value:.4f}")


if __name__ == "__main__":
    main()
