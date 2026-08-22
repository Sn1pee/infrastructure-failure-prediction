"""
Infrastructure Intelligence - Exploratory Data Analysis (EDA) Script
Generates publication-quality figures for telemetry distributions, correlations,
temporal degradation patterns, server-type risk breakdowns, and pre-failure behavior.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set aesthetic styling
sns.set_theme(style="darkgrid")
plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 10})

OUTPUT_DIR = r"C:\Users\monis\.gemini\antigravity\scratch\infrastructure-intelligence\notebooks\eda_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(r"C:\Users\monis\.gemini\antigravity\scratch\infrastructure-intelligence\data\processed\telemetry_cleaned.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])

print(f"Loaded telemetry dataset: {len(df):,} records across {df['server_id'].nunique()} servers.")
print(f"Target Imbalance: Failure=1 -> {df['failure'].sum():,} ({df['failure'].mean():.2%}), Failure=0 -> {(df['failure']==0).sum():,} ({1-df['failure'].mean():.2%})")

# Figure 1: Target Class Imbalance & Server Type Breakdown
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.countplot(data=df, x='failure', palette=['#2ecc71', '#e74c3c'], ax=axes[0])
axes[0].set_title('Target Variable Distribution (Failure vs Normal)', fontsize=12, fontweight='bold')
axes[0].set_xticklabels(['Normal (0)', 'Failure (1)'])
axes[0].set_ylabel('Count')

srv_fail = df.groupby('server_type')['failure'].mean().reset_index()
sns.barplot(data=srv_fail, x='server_type', y='failure', palette='Blues_r', ax=axes[1])
axes[1].set_title('Failure Rate by Server Type', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Failure Rate')
axes[1].set_xlabel('Server Type')
axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '01_target_and_server_breakdown.png'), dpi=300)
plt.close()

# Figure 2: Telemetry Metrics Correlation Heatmap
plt.figure(figsize=(12, 8))
numeric_cols = ['cpu_usage', 'memory_usage', 'disk_usage', 'network_latency', 'packet_loss', 'request_rate', 'error_rate', 'temperature', 'uptime_hours', 'previous_failures', 'failure']
corr = df[numeric_cols].corr()

mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='vlag', vmin=-1, vmax=1, linewidths=0.5)
plt.title('Infrastructure Telemetry Correlation Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '02_correlation_matrix.png'), dpi=300)
plt.close()

# Figure 3: Key Telemetry Metrics Before Failure vs Normal Operation
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

metrics = [
    ('cpu_usage', 'CPU Usage (%)', axes[0, 0]),
    ('memory_usage', 'Memory Usage (%)', axes[0, 1]),
    ('network_latency', 'Network Latency (ms)', axes[1, 0]),
    ('error_rate', 'Error Rate (errors/sec)', axes[1, 1])
]

for col, title, ax in metrics:
    sns.boxplot(data=df, x='failure', y=col, palette=['#3498db', '#e74c3c'], ax=ax, showfliers=False)
    ax.set_title(f'{title} Distribution by Status', fontsize=11, fontweight='bold')
    ax.set_xticklabels(['Normal Operation', 'System Failure'])
    ax.set_xlabel('')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '03_telemetry_distributions_by_failure.png'), dpi=300)
plt.close()

# Figure 4: Hourly System Load & Failure Frequency
hourly_agg = df.groupby(df['timestamp'].dt.hour)[['cpu_usage', 'failure']].agg({'cpu_usage': 'mean', 'failure': 'sum'}).reset_index()

fig, ax1 = plt.subplots(figsize=(12, 5))
ax2 = ax1.twinx()

sns.lineplot(data=hourly_agg, x='timestamp', y='cpu_usage', color='#2980b9', marker='o', ax=ax1, label='Avg CPU Usage (%)')
sns.barplot(data=hourly_agg, x='timestamp', y='failure', color='#e74c3c', alpha=0.4, ax=ax2, label='Total Failures')

ax1.set_xlabel('Hour of Day (00:00 - 23:00)', fontsize=11)
ax1.set_ylabel('Avg CPU Usage (%)', color='#2980b9', fontsize=11)
ax2.set_ylabel('Total Failures', color='#e74c3c', fontsize=11)
plt.title('Diurnal CPU Utilization Pattern vs Failure Counts', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '04_diurnal_cpu_vs_failures.png'), dpi=300)
plt.close()

print(f"EDA visualizations successfully saved to: {OUTPUT_DIR}")
