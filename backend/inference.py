import os
import numpy as np
import pandas as pd
import joblib
from datetime import timedelta

import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")


class SurgePredictor:
    def __init__(self):
        # Load models and metadata
        xgb_path = os.path.join(MODELS_DIR, "xgb_model.pkl")
        lstm_path = os.path.join(MODELS_DIR, "lstm_model.keras")
        ridge_path = os.path.join(MODELS_DIR, "ridge_meta.pkl")
        scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
        meta_path = os.path.join(MODELS_DIR, "meta.npy")

        self.xgb = joblib.load(xgb_path)
        # compile=False skips trying to deserialize loss/metrics
        self.lstm = tf.keras.models.load_model(lstm_path, compile=False)
        self.ridge = joblib.load(ridge_path)
        self.scaler = joblib.load(scaler_path)
        meta = np.load(meta_path, allow_pickle=True).item()
        self.feature_cols = meta["feature_cols"]
        self.seq_len = int(meta["seq_len"])

    def _next_features_row(self, last_row: pd.Series, last_history: pd.DataFrame) -> pd.Series:
        """
        Given the last known row and full history, compute the next-hour feature row.
        """
        # new timestamp
        ts_next = last_row["timestamp"] + timedelta(hours=1)
        hour = ts_next.hour
        dayofweek = ts_next.dayofweek

        # keep AQI and festival flags simple: repeat last values
        aqi = last_row["aqi"]
        is_festival = last_row["is_festival"]

        # build temporary df for rolling features
        # history + last_row as if it is new last point
        temp_hist = last_history.copy()
        temp_hist = pd.concat([temp_hist, last_row.to_frame().T], ignore_index=True)

        adm_lag_1 = temp_hist["admissions"].iloc[-1]
        adm_lag_24 = temp_hist["admissions"].iloc[-24] if len(temp_hist) >= 24 else adm_lag_1
        adm_lag_48 = temp_hist["admissions"].iloc[-48] if len(temp_hist) >= 48 else adm_lag_24

        aqi_lag_1 = temp_hist["aqi"].iloc[-1]
        aqi_lag_24 = temp_hist["aqi"].iloc[-24] if len(temp_hist) >= 24 else aqi_lag_1

        adm_roll_mean_24 = temp_hist["admissions"].tail(24).mean()
        adm_roll_mean_72 = temp_hist["admissions"].tail(72).mean()

        feat = {
            "timestamp": ts_next,
            "hour": hour,
            "dayofweek": dayofweek,
            "aqi": aqi,
            "is_festival": is_festival,
            "adm_lag_1": adm_lag_1,
            "adm_lag_24": adm_lag_24,
            "adm_lag_48": adm_lag_48,
            "aqi_lag_1": aqi_lag_1,
            "aqi_lag_24": aqi_lag_24,
            "adm_roll_mean_24": adm_roll_mean_24,
            "adm_roll_mean_72": adm_roll_mean_72,
        }
        return pd.Series(feat)

    def _supply_plan_from_forecast(self, total_pred: float) -> dict:
        """
        Simple heuristic supply plan based on predicted total admissions.
        """
        # scale factors are arbitrary but give reasonable demo values
        base = max(total_pred, 0.0)

        oxygen_cylinders = int(round(base * 0.15))      # 15% of patients need oxygen
        nebulizers = int(round(base * 0.10))            # 10% need nebulizer use
        burn_dressing_kits = int(round(base * 0.05))    # 5% burn/trauma
        emergency_beds = int(round(base * 0.20))        # 20% need beds reserved

        return {
            "oxygen_cylinders": max(oxygen_cylinders, 0),
            "nebulizer_sets": max(nebulizers, 0),
            "burn_dressing_kits": max(burn_dressing_kits, 0),
            "emergency_beds_to_reserve": max(emergency_beds, 0),
        }

    def predict_horizon(self, horizon: str = "1h") -> dict:
        """
        horizon: "1h", "1d" (24h), "2d" (48h)
        Returns dict with list of {timestamp, predicted_admissions}, and supply plan.
        """
        if horizon == "1h":
            steps = 1
        elif horizon == "1d":
            steps = 24
        elif horizon == "2d":
            steps = 48
        else:
            raise ValueError("Invalid horizon; use '1h', '1d', or '2d'.")

        # Load historical synthetic data
        data_path = os.path.join(os.path.dirname(BASE_DIR), "backend", "data", "synthetic_ed_data.csv")
        # If running from backend/, data_path is fine; if not, use MODELS_DIR base.
        if not os.path.exists(data_path):
            data_path = os.path.join(BASE_DIR, "data", "synthetic_ed_data.csv")

        hist = pd.read_csv(data_path, parse_dates=["timestamp"])

        # Use last N rows as context
        hist = hist.sort_values("timestamp").reset_index(drop=True)
        history = hist.copy()

        # Prepare last feature window for LSTM
        from train_models import create_features  # local import to avoid circular at top

        feat_df = create_features(history)
        feat_df = feat_df.tail(self.seq_len + 1).reset_index(drop=True)
        # last row in feat_df corresponds to last known target; we will use its features as end of sequence context
        X_all = feat_df[self.feature_cols].values
        # last seq_len rows for LSTM sequence
        X_seq_init = X_all[-self.seq_len :, :]
        # scale
        X_seq_scaled = self.scaler.transform(X_seq_init)
        # LSTM expects shape (1, seq_len, n_features)
        lstm_seq = X_seq_scaled.reshape(1, self.seq_len, X_seq_scaled.shape[1])

        # last_row for feature evolution
        last_row = history.iloc[-1].copy()

        results = []

        for _ in range(steps):
            # build next feature row
            next_feat_row = self._next_features_row(last_row, history)
            # append to history for next iteration
            history = pd.concat([history, next_feat_row.to_frame().T], ignore_index=True)

            # XGBoost features vector
            x_vec = next_feat_row[self.feature_cols].values.reshape(1, -1)
            x_vec_scaled = self.scaler.transform(x_vec)
            xgb_pred = self.xgb.predict(x_vec_scaled)[0]

            # update LSTM sequence: drop oldest, append new scaled features
            lstm_seq_features = np.concatenate(
                [lstm_seq[:, 1:, :], x_vec_scaled.reshape(1, 1, -1)],
                axis=1,
            )
            lstm_pred = self.lstm.predict(lstm_seq_features, verbose=0).flatten()[0]

            # update sequence state
            lstm_seq = lstm_seq_features

            # ensemble prediction
            meta_X = np.array([[xgb_pred, lstm_pred]])
            ens_pred = float(self.ridge.predict(meta_X)[0])

            # set last_row admissions to ensemble prediction for next step's lags/rolling
            last_row["timestamp"] = next_feat_row["timestamp"]
            last_row["hour"] = next_feat_row["hour"]
            last_row["dayofweek"] = next_feat_row["dayofweek"]
            last_row["aqi"] = next_feat_row["aqi"]
            last_row["is_festival"] = next_feat_row["is_festival"]
            last_row["admissions"] = max(ens_pred, 0.0)

            results.append(
                {
                    "timestamp": next_feat_row["timestamp"].isoformat(),
                    "predicted_admissions": max(ens_pred, 0.0),
                    "xgb_only": max(float(xgb_pred), 0.0),
                    "lstm_only": max(float(lstm_pred), 0.0),
                }
            )

        total_pred = sum(r["predicted_admissions"] for r in results)
        supply_plan = self._supply_plan_from_forecast(total_pred)

        return {
            "horizon": horizon,
            "steps": steps,
            "total_predicted_admissions": total_pred,
            "forecast": results,
            "supply_plan": supply_plan,
        }
