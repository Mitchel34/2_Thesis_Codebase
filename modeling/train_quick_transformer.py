#!/usr/bin/env python3
"""
Quick CPU-friendly baseline: simple residual predictor over hourly sequences.
- Input features: [nwm_cms, selected ERA5 cols if present] per hour
- Target: y_residual_cms (USGS - NWM)
- Train: first 6 days; Test: last day
"""
import os
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Minimal set of features to avoid missing columns
ERA5_CANDIDATES = [
    'temp_c', 'dewpoint_c', 'pressure_hpa', 'precip_mm', 'radiation_mj_m2',
    'wind_speed', 'vpd_kpa', 'rel_humidity_pct', 'soil_moisture_vwc',
    'hour_sin', 'hour_cos', 'doy_sin', 'doy_cos', 'month_sin', 'month_cos'
]


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    # Ensure required columns exist
    assert {'timestamp','nwm_cms','usgs_cms','y_residual_cms'}.issubset(df.columns), "Missing core columns"
    # Sort and drop any NaNs in target
    df = df.sort_values('timestamp').copy()
    df = df.dropna(subset=['y_residual_cms', 'nwm_cms', 'usgs_cms'])
    return df


def make_feature_matrix(df: pd.DataFrame):
    features = ['nwm_cms'] + [c for c in ERA5_CANDIDATES if c in df.columns]
    X = df[features].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = df['y_residual_cms'].values.astype('float32')
    return X.values.astype('float32'), y, features


def to_sequences(X: np.ndarray, y: np.ndarray, seq_len: int = 24):
    # Create sliding windows of length seq_len to predict next-step residual
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_len):
        X_seq.append(X[i:i+seq_len])
        y_seq.append(y[i+seq_len])
    return np.array(X_seq), np.array(y_seq)


def build_model(seq_len: int, n_features: int):
    inp = keras.Input(shape=(seq_len, n_features))
    # Lightweight transformer block
    x = layers.LayerNormalization()(inp)
    x = layers.MultiHeadAttention(num_heads=2, key_dim=min(16, n_features))(x, x)
    x = layers.Dropout(0.1)(x)
    x = layers.Add()([inp, x])
    x = layers.LayerNormalization()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.GlobalAveragePooling1D()(x)
    out = layers.Dense(1)(x)
    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss='mse')
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='Path to Parquet dataset')
    ap.add_argument('--seq-len', type=int, default=24)
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--batch-size', type=int, default=32)
    args = ap.parse_args()

    df = load_data(args.data)
    # Split 6 days train, 1 day test
    start = df['timestamp'].min()
    split_time = start + pd.Timedelta(days=6)
    train = df[df['timestamp'] < split_time]
    test = df[df['timestamp'] >= split_time]

    X_train_raw, y_train, feat = make_feature_matrix(train)
    X_test_raw, y_test, _ = make_feature_matrix(test)

    seq_len = args.seq_len
    X_train, y_train = to_sequences(X_train_raw, y_train, seq_len)
    X_test, y_test = to_sequences(X_test_raw, y_test, seq_len)

    if len(X_train) == 0 or len(X_test) == 0:
        print('Not enough data for sequences; reduce seq-len or extend window.')
        return

    model = build_model(seq_len, X_train.shape[-1])
    model.summary()

    cb = [keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True, monitor='val_loss')]
    hist = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=1,
        callbacks=cb,
    )

    # Evaluate (MSE and simple RMSE)
    pred = model.predict(X_test, verbose=0).squeeze()
    mse = np.mean((pred - y_test)**2)
    rmse = np.sqrt(mse)
    print(f"Test RMSE on residual: {rmse:.3f} (MSE={mse:.3f})")

    # Optional: write simple outputs
    out_dir = 'data/clean/modeling'
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, 'quick_pred.npy'), pred)
    np.save(os.path.join(out_dir, 'quick_true.npy'), y_test)
    with open(os.path.join(out_dir, 'quick_features.txt'), 'w') as f:
        f.write("\n".join(feat))


if __name__ == '__main__':
    main()
