-- Infrastructure Intelligence Database Schema
-- Table: telemetry

DROP TABLE IF EXISTS telemetry;

CREATE TABLE telemetry (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME NOT NULL,
    server_id VARCHAR(20) NOT NULL,
    server_type VARCHAR(50) NOT NULL,
    region VARCHAR(50) NOT NULL,
    cpu_usage REAL NOT NULL,
    memory_usage REAL NOT NULL,
    disk_usage REAL NOT NULL,
    network_latency REAL NOT NULL,
    packet_loss REAL NOT NULL,
    request_rate REAL NOT NULL,
    error_rate REAL NOT NULL,
    active_connections INTEGER NOT NULL,
    temperature REAL NOT NULL,
    uptime_hours REAL NOT NULL,
    workload_intensity REAL NOT NULL,
    previous_failures INTEGER NOT NULL,
    maintenance_flag INTEGER NOT NULL,
    failure INTEGER NOT NULL
);

-- Indexes for analytical performance
CREATE INDEX idx_telemetry_server_id ON telemetry(server_id);
CREATE INDEX idx_telemetry_timestamp ON telemetry(timestamp);
CREATE INDEX idx_telemetry_failure ON telemetry(failure);
CREATE INDEX idx_telemetry_server_type ON telemetry(server_type);
CREATE INDEX idx_telemetry_region ON telemetry(region);
