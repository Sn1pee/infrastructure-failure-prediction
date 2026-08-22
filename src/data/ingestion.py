"""
Infrastructure Intelligence - Data Ingestion & Synthetic Telemetry Generator
Generates realistic IT infrastructure telemetry data with non-linear correlations
and temporal properties for server failure prediction & anomaly detection.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
import os


class TelemetryGenerator:
    """Generates synthetic multi-server infrastructure telemetry dataset with realistic coupling."""

    SERVER_TYPES = ['Database', 'Web Server', 'App Server', 'Storage', 'Compute Node']
    REGIONS = ['us-east-1', 'us-west-2', 'eu-central-1', 'ap-southeast-1']

    def __init__(self, num_servers: int = 100, hours: int = 600, random_seed: int = 42):
        self.num_servers = num_servers
        self.hours = hours
        self.random_seed = random_seed

    def generate(self) -> pd.DataFrame:
        """Generate telemetry DataFrame with realistic physics and correlations."""
        np.random.seed(self.random_seed)
        start_time = datetime(2026, 1, 1, 0, 0, 0)

        servers = [f"SRV-{i+1:03d}" for i in range(self.num_servers)]
        server_meta = {
            srv: {
                'type': np.random.choice(self.SERVER_TYPES),
                'region': np.random.choice(self.REGIONS),
                'base_cpu': np.random.uniform(25, 45),
                'base_mem': np.random.uniform(30, 50),
                'base_disk': np.random.uniform(40, 70),
                'health_bias': np.random.uniform(-0.5, 0.5)
            }
            for srv in servers
        }

        data_rows = []

        for srv_idx, srv in enumerate(servers):
            meta = server_meta[srv]
            prev_failures = 0
            uptime = np.random.uniform(10, 500)

            for h in range(self.hours):
                current_time = start_time + timedelta(hours=h)
                hour_of_day = current_time.hour
                day_of_week = current_time.weekday()

                # Diurnal load cycle (higher during business hours 8-18)
                diurnal = 0.3 * np.sin(2 * np.pi * (hour_of_day - 8) / 24) + (0.1 if day_of_week < 5 else -0.15)
                workload_intensity = np.clip(1.0 + diurnal + np.random.normal(0, 0.15), 0.4, 2.5)

                # Request rate & active connections scale with workload
                base_req = 1500 if meta['type'] in ['Web Server', 'App Server'] else 600
                request_rate = np.clip(base_req * workload_intensity + np.random.normal(0, 100), 50, 6000)
                active_connections = np.clip(request_rate * 0.4 + np.random.normal(0, 30), 10, 3000)

                # CPU & Memory load
                cpu_usage = np.clip(meta['base_cpu'] * workload_intensity + np.random.normal(0, 8), 5, 99.9)
                memory_usage = np.clip(meta['base_mem'] * (1 + 0.4 * diurnal) + (cpu_usage * 0.25) + np.random.normal(0, 5), 10, 99.9)

                # Disk usage slowly increases with time + workload
                disk_usage = np.clip(meta['base_disk'] + (h * 0.03) + np.random.normal(0, 0.5), 15, 99.0)

                # Temperature correlated with CPU load
                temperature = np.clip(32.0 + 0.48 * cpu_usage + np.random.normal(0, 2.5), 30.0, 98.0)

                # Latency & packet loss coupling
                latency_base = 15 if meta['type'] != 'Database' else 25
                latency_spike = (50 if cpu_usage > 85 else 0) + (80 if memory_usage > 90 else 0)
                network_latency = np.clip(latency_base + (request_rate / 80) + latency_spike + np.random.normal(0, 5), 5.0, 450.0)

                packet_loss = np.clip(np.maximum(0, (network_latency - 60) / 30) + np.random.normal(0, 0.3), 0.0, 15.0)

                # Error rate rises with high latency and CPU overload
                error_base = 0.5
                if cpu_usage > 90 or memory_usage > 92 or network_latency > 150:
                    error_base += np.random.uniform(10, 80)
                error_rate = np.clip(error_base + np.random.normal(0, 2), 0.0, 200.0)

                # Maintenance events occur periodically or after extreme load
                maintenance_flag = 1 if (h > 0 and h % 168 == 0 and np.random.rand() < 0.3) else 0
                if maintenance_flag == 1:
                    uptime = 0.0
                else:
                    uptime += 1.0

                # Failure log-odds calculation (imbalanced target ~3-5%)
                z = (
                    -4.3
                    + 0.048 * (cpu_usage - 55)
                    + 0.042 * (memory_usage - 60)
                    + 0.035 * (temperature - 65)
                    + 0.018 * (network_latency - 50)
                    + 0.045 * error_rate
                    + 0.35 * prev_failures
                    + meta['health_bias']
                    - 2.8 * maintenance_flag
                )
                prob_failure = 1.0 / (1.0 + np.exp(-z))
                failure = 1 if (np.random.rand() < prob_failure and maintenance_flag == 0) else 0

                if failure == 1:
                    prev_failures += 1
                    uptime = 0.0  # reset uptime after failure

                data_rows.append({
                    'timestamp': current_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'server_id': srv,
                    'server_type': meta['type'],
                    'region': meta['region'],
                    'cpu_usage': round(float(cpu_usage), 2),
                    'memory_usage': round(float(memory_usage), 2),
                    'disk_usage': round(float(disk_usage), 2),
                    'network_latency': round(float(network_latency), 2),
                    'packet_loss': round(float(packet_loss), 2),
                    'request_rate': round(float(request_rate), 2),
                    'error_rate': round(float(error_rate), 2),
                    'active_connections': int(active_connections),
                    'temperature': round(float(temperature), 2),
                    'uptime_hours': round(float(uptime), 1),
                    'workload_intensity': round(float(workload_intensity), 3),
                    'previous_failures': int(prev_failures),
                    'maintenance_flag': int(maintenance_flag),
                    'failure': int(failure)
                })

        df = pd.DataFrame(data_rows)
        return df


def generate_and_save_raw_data(
    output_path: str = r"C:\Users\monis\.gemini\antigravity\scratch\infrastructure-intelligence\data\raw\telemetry_raw.csv",
    num_servers: int = 100,
    hours: int = 600
) -> pd.DataFrame:
    """Generate telemetry dataset and save raw CSV."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    generator = TelemetryGenerator(num_servers=num_servers, hours=hours, random_seed=42)
    df = generator.generate()
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df):,} telemetry records across {num_servers} servers ({hours} hours).")
    print(f"Target failure rate: {df['failure'].mean():.2%}")
    print(f"Saved raw dataset to: {output_path}")
    return df


if __name__ == "__main__":
    generate_and_save_raw_data()
