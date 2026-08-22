"""
Infrastructure Intelligence - Feature Engineering Pipeline
Creates temporal, rolling-window time-series metrics, interaction stress indices,
and categorical encodings while preventing data leakage across temporal boundaries.
"""

import pandas as pd
import numpy as np
import os
from typing import Tuple, List, Dict, Any
from sklearn.preprocessing import OneHotEncoder
import joblib


class FeatureEngineer:
    """Extracts domain-specific infrastructure metrics & handles temporal splits."""

    CATEGORICAL_COLS = ['server_type', 'region']

    NUMERICAL_COLS = [
        'cpu_usage', 'memory_usage', 'disk_usage', 'network_latency',
        'packet_loss', 'request_rate', 'error_rate', 'active_connections',
        'temperature', 'uptime_hours', 'workload_intensity',
        'previous_failures', 'maintenance_flag'
    ]

    def __init__(self):
        self.encoder: OneHotEncoder = None
        self.feature_names: List[str] = []

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction terms, time features, and rolling time-series metrics."""
        df = df.copy()

        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Sort strictly by server and timestamp
        df = df.sort_values(by=['server_id', 'timestamp']).reset_index(drop=True)

        # 1. Temporal Time Features
        df['hour_of_day'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

        # 2. Infrastructure Interaction Terms
        df['cpu_memory_stress'] = df['cpu_usage'] * df['memory_usage'] / 100.0
        df['network_stress_score'] = df['network_latency'] * (1.0 + df['packet_loss'])
        df['system_utilization_index'] = (df['cpu_usage'] + df['memory_usage'] + df['disk_usage']) / 3.0
        df['error_to_request_ratio'] = df['error_rate'] / (df['request_rate'] + 1.0)
        df['thermal_efficiency_delta'] = df['temperature'] - (30.0 + 0.45 * df['cpu_usage'])
        df['conn_per_request'] = df['active_connections'] / (df['request_rate'] + 1.0)

        # 3. Rolling Window Time-Series Statistics per Server
        # We shift by 1 or compute rolling statistics on historical sequence to avoid target leakage
        grouped = df.groupby('server_id')

        # 3-Hour Rolling Features
        df['cpu_roll_mean_3h'] = grouped['cpu_usage'].transform(lambda x: x.rolling(3, min_periods=1).mean())
        df['cpu_roll_std_3h'] = grouped['cpu_usage'].transform(lambda x: x.rolling(3, min_periods=1).std()).fillna(0.0)

        df['memory_roll_mean_3h'] = grouped['memory_usage'].transform(lambda x: x.rolling(3, min_periods=1).mean())
        df['latency_roll_mean_3h'] = grouped['network_latency'].transform(lambda x: x.rolling(3, min_periods=1).mean())
        df['error_roll_mean_3h'] = grouped['error_rate'].transform(lambda x: x.rolling(3, min_periods=1).mean())

        # 6-Hour Rolling Features
        df['cpu_roll_mean_6h'] = grouped['cpu_usage'].transform(lambda x: x.rolling(6, min_periods=1).mean())
        df['memory_roll_mean_6h'] = grouped['memory_usage'].transform(lambda x: x.rolling(6, min_periods=1).mean())
        df['error_roll_mean_6h'] = grouped['error_rate'].transform(lambda x: x.rolling(6, min_periods=1).mean())

        # Fill any remaining NaNs in rolling features
        rolling_cols = [c for c in df.columns if 'roll_' in c]
        df[rolling_cols] = df[rolling_cols].fillna(0.0)

        return df

    def fit_transform_categorical(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Fit OneHotEncoder on categorical columns and return encoded DataFrame."""
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded_array = self.encoder.fit_transform(df[self.CATEGORICAL_COLS])
        encoded_col_names = self.encoder.get_feature_names_out(self.CATEGORICAL_COLS).tolist()

        encoded_df = pd.DataFrame(encoded_array, columns=encoded_col_names, index=df.index)
        return encoded_df, encoded_col_names

    def transform_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform categorical columns using fitted OneHotEncoder."""
        if self.encoder is None:
            raise ValueError("Encoder has not been fitted. Call fit_transform_categorical first.")
        encoded_array = self.encoder.transform(df[self.CATEGORICAL_COLS])
        encoded_col_names = self.encoder.get_feature_names_out(self.CATEGORICAL_COLS).tolist()
        return pd.DataFrame(encoded_array, columns=encoded_col_names, index=df.index)

    def prepare_dataset(
        self,
        df: pd.DataFrame,
        is_train: bool = True
    ) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """Complete feature generation pipeline producing feature matrix X and target y."""
        df_feat = self.create_features(df)

        if is_train:
            encoded_df, cat_feature_names = self.fit_transform_categorical(df_feat)
        else:
            encoded_df = self.transform_categorical(df_feat)
            cat_feature_names = self.encoder.get_feature_names_out(self.CATEGORICAL_COLS).tolist()

        # Combine numerical, interaction, rolling, and categorical features
        output_score_cols = [
            'failure_probability', 'risk_score', 'risk_level', 'predicted_failure',
            'anomaly_score', 'is_anomaly'
        ]
        exclude_cols = ['timestamp', 'server_id', 'failure'] + output_score_cols + self.CATEGORICAL_COLS
        feature_cols = [col for col in df_feat.columns if col not in exclude_cols]

        X_num = df_feat[feature_cols].reset_index(drop=True)
        X = pd.concat([X_num, encoded_df.reset_index(drop=True)], axis=1)
        y = df_feat['failure'].reset_index(drop=True) if 'failure' in df_feat.columns else pd.Series(0, index=df_feat.index)

        if is_train or not self.feature_names:
            self.feature_names = X.columns.tolist()
        else:
            # Ensure X contains exact features in exact order expected by scaler/model
            for col in self.feature_names:
                if col not in X.columns:
                    X[col] = 0.0
            X = X[self.feature_names]

        return X, y, self.feature_names


def temporal_train_val_test_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataset chronologically based on timestamp to avoid temporal data leakage."""
    df = df.sort_values('timestamp').reset_index(drop=True)

    unique_times = df['timestamp'].sort_values().unique()
    n_times = len(unique_times)

    train_idx_end = int(n_times * train_ratio)
    val_idx_end = int(n_times * (train_ratio + val_ratio))

    train_cutoff = unique_times[train_idx_end]
    val_cutoff = unique_times[val_idx_end]

    train_df = df[df['timestamp'] < train_cutoff].copy()
    val_df = df[(df['timestamp'] >= train_cutoff) & (df['timestamp'] < val_cutoff)].copy()
    test_df = df[df['timestamp'] >= val_cutoff].copy()

    print(f"Temporal Split Summary:")
    print(f"  Train Set: {len(train_df):,} rows (failures: {train_df['failure'].sum()} / {train_df['failure'].mean():.2%})")
    print(f"  Val Set  : {len(val_df):,} rows (failures: {val_df['failure'].sum()} / {val_df['failure'].mean():.2%})")
    print(f"  Test Set : {len(test_df):,} rows (failures: {test_df['failure'].sum()} / {test_df['failure'].mean():.2%})")

    return train_df, val_df, test_df


def run_feature_engineering_pipeline(
    cleaned_csv_path: str = r"C:\Users\monis\.gemini\antigravity\scratch\infrastructure-intelligence\data\processed\telemetry_cleaned.csv",
    output_features_path: str = r"C:\Users\monis\.gemini\antigravity\scratch\infrastructure-intelligence\data\processed\features.csv",
    scaler_out_path: str = r"C:\Users\monis\.gemini\antigravity\scratch\infrastructure-intelligence\models\feature_engineer.pkl"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, FeatureEngineer]:
    """Run full feature engineering and temporal split pipeline."""
    print("--- Starting Feature Engineering Pipeline ---")
    df = pd.read_csv(cleaned_csv_path)

    fe = FeatureEngineer()
    df_featured = fe.create_features(df)

    # Save complete featured dataset
    os.makedirs(os.path.dirname(output_features_path), exist_ok=True)
    df_featured.to_csv(output_features_path, index=False)
    print(f"Saved featured telemetry dataset ({df_featured.shape[1]} columns) to: {output_features_path}")

    # Perform temporal split
    train_df, val_df, test_df = temporal_train_val_test_split(df_featured)

    # Fit encoder on train, transform train/val/test
    X_train, y_train, feat_names = fe.prepare_dataset(train_df, is_train=True)
    X_val, y_val, _ = fe.prepare_dataset(val_df, is_train=False)
    X_test, y_test, _ = fe.prepare_dataset(test_df, is_train=False)

    # Save feature engineer model pipeline
    os.makedirs(os.path.dirname(scaler_out_path), exist_ok=True)
    joblib.dump(fe, scaler_out_path)
    print(f"Saved FeatureEngineer encoder pipeline to: {scaler_out_path}")

    return train_df, val_df, test_df, df_featured, fe


if __name__ == "__main__":
    run_feature_engineering_pipeline()
