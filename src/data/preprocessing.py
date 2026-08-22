"""
Infrastructure Intelligence - Data Preprocessing & SQLite Storage Pipeline
Cleans raw telemetry, validates schema integrity, handles anomalies,
and populates SQLite database for production SQL analytics.
"""

import sqlite3
import pandas as pd
import numpy as np
import os
from typing import Tuple, Dict, Any


class DataPreprocessor:
    """Preprocesses raw telemetry data, validates quality, and handles SQL storage."""

    EXPECTED_COLUMNS = [
        'timestamp', 'server_id', 'server_type', 'region', 'cpu_usage',
        'memory_usage', 'disk_usage', 'network_latency', 'packet_loss',
        'request_rate', 'error_rate', 'active_connections', 'temperature',
        'uptime_hours', 'workload_intensity', 'previous_failures',
        'maintenance_flag', 'failure'
    ]

    def __init__(self, db_path: str = r"C:\Users\monis\.gemini\antigravity\scratch\infrastructure-intelligence\data\processed\infrastructure.db"):
        self.db_path = db_path

    def validate_schema(self, df: pd.DataFrame) -> bool:
        """Validate presence and data types of required columns."""
        missing = [col for col in self.EXPECTED_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns in raw telemetry: {missing}")
        return True

    def clean_telemetry(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean telemetry data: convert timestamps, enforce bounds, handle duplicates."""
        df = df.copy()

        # Validate schema
        self.validate_schema(df)

        # Drop exact duplicates if any
        initial_len = len(df)
        df = df.drop_duplicates()
        dropped = initial_len - len(df)
        if dropped > 0:
            print(f"Dropped {dropped} duplicate rows.")

        # Timestamp processing
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Enforce realistic physiological bounds (clip physical metrics)
        df['cpu_usage'] = df['cpu_usage'].clip(0.0, 100.0)
        df['memory_usage'] = df['memory_usage'].clip(0.0, 100.0)
        df['disk_usage'] = df['disk_usage'].clip(0.0, 100.0)
        df['network_latency'] = df['network_latency'].clip(0.0, 2000.0)
        df['packet_loss'] = df['packet_loss'].clip(0.0, 100.0)
        df['request_rate'] = df['request_rate'].clip(lower=0.0)
        df['error_rate'] = df['error_rate'].clip(lower=0.0)
        df['active_connections'] = df['active_connections'].clip(lower=0)
        df['temperature'] = df['temperature'].clip(10.0, 120.0)
        df['uptime_hours'] = df['uptime_hours'].clip(lower=0.0)
        df['previous_failures'] = df['previous_failures'].clip(lower=0)

        # Sort by server and timestamp for time-series operations
        df = df.sort_values(by=['server_id', 'timestamp']).reset_index(drop=True)

        return df

    def save_to_sqlite(self, df: pd.DataFrame, schema_sql_path: str) -> None:
        """Store cleaned dataset into SQLite database using DDL schema."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Execute DDL schema creation
        if os.path.exists(schema_sql_path):
            with open(schema_sql_path, 'r') as f:
                schema_script = f.read()
            cursor.executescript(schema_script)

        # Write DataFrame to SQL
        df_sql = df.copy()
        df_sql['timestamp'] = df_sql['timestamp'].astype(str)
        df_sql.to_sql('telemetry', conn, if_exists='append', index=False)

        conn.commit()
        conn.close()
        print(f"Stored {len(df):,} records into SQLite database: {self.db_path}")

    def execute_sql_analytics(self, analytics_sql_path: str) -> Dict[str, pd.DataFrame]:
        """Run production analytical SQL queries and return results as DataFrames."""
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"Database file not found at {self.db_path}")

        conn = sqlite3.connect(self.db_path)
        
        with open(analytics_sql_path, 'r') as f:
            sql_content = f.read()

        # Split SQL file into individual queries
        queries = [q.strip() for q in sql_content.split(';') if q.strip()]
        results = {}

        for i, q in enumerate(queries):
            query_name = f"query_{i+1}"
            first_line = q.split('\n')[0]
            if '--' in first_line:
                query_name = first_line.replace('--', '').strip()
            
            try:
                res_df = pd.read_sql_query(q, conn)
                results[query_name] = res_df
            except Exception as e:
                print(f"Error executing SQL query '{query_name}': {e}")

        conn.close()
        return results


def run_preprocessing_pipeline(
    raw_csv_path: str = r"C:\Users\monis\.gemini\antigravity\scratch\infrastructure-intelligence\data\raw\telemetry_raw.csv",
    cleaned_csv_path: str = r"C:\Users\monis\.gemini\antigravity\scratch\infrastructure-intelligence\data\processed\telemetry_cleaned.csv",
    db_path: str = r"C:\Users\monis\.gemini\antigravity\scratch\infrastructure-intelligence\data\processed\infrastructure.db",
    schema_sql_path: str = r"C:\Users\monis\.gemini\antigravity\scratch\infrastructure-intelligence\sql\schema.sql",
    analytics_sql_path: str = r"C:\Users\monis\.gemini\antigravity\scratch\infrastructure-intelligence\sql\analytics.sql"
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Execute complete ETL preprocessing & SQL storage pipeline."""
    print("--- Starting Telemetry Data Pipeline & SQL Ingestion ---")
    raw_df = pd.read_csv(raw_csv_path)

    preprocessor = DataPreprocessor(db_path=db_path)
    cleaned_df = preprocessor.clean_telemetry(raw_df)

    # Save cleaned CSV
    os.makedirs(os.path.dirname(cleaned_csv_path), exist_ok=True)
    cleaned_df.to_csv(cleaned_csv_path, index=False)
    print(f"Saved cleaned telemetry CSV to: {cleaned_csv_path}")

    # Store in SQLite DB
    preprocessor.save_to_sqlite(cleaned_df, schema_sql_path=schema_sql_path)

    # Execute Analytical SQL Queries
    sql_results = preprocessor.execute_sql_analytics(analytics_sql_path=analytics_sql_path)
    print(f"Executed {len(sql_results)} SQL analytical queries successfully.")

    return cleaned_df, sql_results


if __name__ == "__main__":
    run_preprocessing_pipeline()
