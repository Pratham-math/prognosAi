import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# add this line near other imports
from keras.saving import save_model

import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

from xgboost import XGBRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import joblib


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


def generate_synthetic_data(n_days: int = 365 * 2) -> pd.DataFrame:
    """
    Generate hourly synthetic ED admissions with:
    - daily pattern (night low, evening higher)
    - weekend effect
    - a few 'festival' windows with higher load
    - AQI feature correlated with admissions
    """
    hours = n_days * 24
    idx = pd.date_range(start="2023-01-01", periods=hours, freq="h")  # lowercase 'h'

    df = pd.DataFrame({"timestamp": idx})

    # Base rate around 5 patients / hour
    base_rate = 5.0

    # Hour-of-day pattern: sinusoid
    hour = df["timestamp"].dt.hour
    hour_factor = 1.0 + 0.5 * np.sin(2 * np.pi * (hour - 12) / 24.0)

    # Weekend factor
    dow = df["timestamp"].dt.dayofweek
    weekend_factor = np.where(dow >= 5, 1.2, 1.0)

    # Synthetic AQI (pollution) feature
    # Base around 100, some slow variation + noise
    t = np.arange(hours)
    aqi = 100 + 40 * np.sin(2 * np.pi * t / (24 * 14)) + 20 * np.random.randn(hours)

    # A few 'festival windows' with spikes in AQI and admissions
    festival_flag = np.zeros(hours)
    for start_day in [60, 180, 300]:
        start_idx = start_day * 24
        end_idx = start_idx + 72  # 3 days
        if end_idx > hours:
            break
        aqi[start_idx:end_idx] += 80
        festival_flag[start_idx:end_idx] = 1.0

    # Expected admissions (lambda for Poisson)
    expected = base_rate * hour_factor * weekend_factor
    # Pollution effect (only above 100)
    expected *= 1.0 + 0.003 * np.clip(aqi - 100, 0, None)
    # Festival effect
    expected *= 1.0 + 0.4 * festival_flag

    # Actual admissions as Poisson draw
    admissions = np.random.poisson(lam=np.clip(expected, 0.1, None))

    df["hour"] = hour
    df["dayofweek"] = dow
    df["aqi"] = aqi
    df["is_festival"] = festival_flag
    df["admissions"] = admissions

    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create lag and rolling-window features and 1-step-ahead target.
    Target: admissions at t+1 hour.
    """
    df = df.copy()

    # Simple lags of admissions and AQI
    df["adm_lag_1"] = df["admissions"].shift(1)
    df["adm_lag_24"] = df["admissions"].shift(24)
    df["adm_lag_48"] = df["admissions"].shift(48)

    df["aqi_lag_1"] = df["aqi"].shift(1)
    df["aqi_lag_24"] = df["aqi"].shift(24)

    # Rolling means of admissions
    df["adm_roll_mean_24"] = df["admissions"].rolling(24).mean()
    df["adm_roll_mean_72"] = df["admissions"].rolling(72).mean()

    # Target: next-hour admissions
    df["target"] = df["admissions"].shift(-1)

    # Drop rows with NaNs from lags/rolling/target
    df = df.dropna().reset_index(drop=True)
    return df


def make_sequences(X: np.ndarray, y: np.ndarray, seq_len: int = 48):
    """
    Turn tabular features into sequences for LSTM.
    X: (n_samples, n_features)
    y: (n_samples,)
    Returns:
      X_seq: (n_seq, seq_len, n_features)
      y_seq: (n_seq,)
    """
    X_seq = []
    y_seq = []
    for i in range(seq_len, len(X)):
        X_seq.append(X[i - seq_len : i, :])
        y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq)


def build_lstm_model(input_seq_len: int, n_features: int):
    model = Sequential()
    model.add(
        LSTM(
            64,
            return_sequences=True,
            input_shape=(input_seq_len, n_features),
        )
    )
    model.add(Dropout(0.3))
    model.add(LSTM(32))
    model.add(Dropout(0.3))
    model.add(Dense(16, activation="relu"))
    model.add(Dense(1))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"],
    )
    return model


def main():
    print("Generating synthetic data...")
    df_raw = generate_synthetic_data()
    csv_path = os.path.join(DATA_DIR, "synthetic_ed_data.csv")
    df_raw.to_csv(csv_path, index=False)
    print(f"Synthetic data saved to {csv_path}")

    print("Creating features...")
    df = create_features(df_raw)

    feature_cols = [
        "hour",
        "dayofweek",
        "aqi",
        "is_festival",
        "adm_lag_1",
        "adm_lag_24",
        "adm_lag_48",
        "aqi_lag_1",
        "aqi_lag_24",
        "adm_roll_mean_24",
        "adm_roll_mean_72",
    ]
    target_col = "target"

    X = df[feature_cols].values
    y = df[target_col].values

    # Chronological split: 80% train, 20% test
    split_idx = int(len(X) * 0.8)
    X_train_raw, X_test_raw = X[:split_idx], X[split_idx:]
    y_train_raw, y_test_raw = y[:split_idx], y[split_idx:]

    print(f"Train samples: {len(X_train_raw)}, Test samples: {len(X_test_raw)}")

    # Scale features
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    # -----------------------
    # XGBoost base model
    # -----------------------
    print("Training XGBoost...")
    xgb = XGBRegressor(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        n_jobs=-1,
        random_state=42,
    )
    xgb.fit(X_train, y_train_raw)
    xgb_test_pred = xgb.predict(X_test)

    xgb_mae = mean_absolute_error(y_test_raw, xgb_test_pred)
    xgb_rmse = np.sqrt(mean_squared_error(y_test_raw, xgb_test_pred))
    print(f"XGBoost Test MAE:  {xgb_mae:.3f}")
    print(f"XGBoost Test RMSE: {xgb_rmse:.3f}")

    # -----------------------
    # LSTM base model
    # -----------------------
    print("Preparing sequences for LSTM...")
    SEQ_LEN = 48  # 2 days of history

    X_train_seq, y_train_seq = make_sequences(X_train, y_train_raw, seq_len=SEQ_LEN)
    X_test_seq, y_test_seq = make_sequences(X_test, y_test_raw, seq_len=SEQ_LEN)

    print(f"LSTM train seq shape: {X_train_seq.shape}")
    print(f"LSTM test  seq shape: {X_test_seq.shape}")

    lstm = build_lstm_model(input_seq_len=SEQ_LEN, n_features=X_train_seq.shape[2])

    early_stop = EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5
    )

    print("Training LSTM...")
    lstm.fit(
        X_train_seq,
        y_train_seq,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stop, reduce_lr],
        verbose=1,
    )

    lstm_test_pred = lstm.predict(X_test_seq).flatten()
    lstm_mae = mean_absolute_error(y_test_seq, lstm_test_pred)
    lstm_rmse = np.sqrt(mean_squared_error(y_test_seq, lstm_test_pred))
    print(f"LSTM Test MAE:  {lstm_mae:.3f}")
    print(f"LSTM Test RMSE: {lstm_rmse:.3f}")

    # -----------------------
    # Ridge ensemble meta-learner
    # -----------------------
    print("Training Ridge ensemble...")
    # Align XGBoost test predictions with LSTM test targets.
    # LSTM test y starts SEQ_LEN steps into the test segment.
    xgb_test_pred_for_meta = xgb_test_pred[SEQ_LEN:]
    y_meta = y_test_seq  # same length as LSTM predictions
    assert len(xgb_test_pred_for_meta) == len(lstm_test_pred) == len(y_meta)

    meta_X = np.column_stack([xgb_test_pred_for_meta, lstm_test_pred])
    ridge = Ridge(alpha=1.0)
    ridge.fit(meta_X, y_meta)

    ensemble_pred = ridge.predict(meta_X)
    ens_mae = mean_absolute_error(y_meta, ensemble_pred)
    ens_rmse = np.sqrt(mean_squared_error(y_meta, ensemble_pred))

    print("ENSEMBLE RESULTS")
    print(f"  Ensemble Test MAE:  {ens_mae:.3f}")
    print(f"  Ensemble Test RMSE: {ens_rmse:.3f}")
    print(f"  XGB-only  MAE (same subset): {mean_absolute_error(y_meta, xgb_test_pred_for_meta):.3f}")
    print(f"  LSTM-only MAE (same subset): {mean_absolute_error(y_meta, lstm_test_pred):.3f}")
    print("  Ridge weights: XGB = {:.3f}, LSTM = {:.3f}".format(ridge.coef_[0], ridge.coef_[1]))

    # -----------------------
    # Save models and metadata
    # -----------------------
    print("Saving models and scaler...")
    joblib.dump(xgb, os.path.join(MODELS_DIR, "xgb_model.pkl"))

    # Save LSTM in new Keras format to avoid HDF5 deserialization issues
    lstm_path = os.path.join(MODELS_DIR, "lstm_model.keras")
    tf.keras.models.save_model(lstm, lstm_path)

    joblib.dump(ridge, os.path.join(MODELS_DIR, "ridge_meta.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))


    # Save metadata (feature columns and sequence length)
    meta = {
        "feature_cols": feature_cols,
        "seq_len": SEQ_LEN,
    }
    meta_path = os.path.join(MODELS_DIR, "meta.npy")
    np.save(meta_path, meta, allow_pickle=True)

    print("All models and metadata saved to 'models/'.")


if __name__ == "__main__":
    main()
