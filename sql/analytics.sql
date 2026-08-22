-- Infrastructure Intelligence - Analytical SQL Queries
-- Purpose: Extract telemetry insights, pre-failure metrics, and server risk breakdowns.

-- 1. Failure Rate by Server Type
SELECT 
    server_type,
    COUNT(*) AS total_observations,
    SUM(failure) AS total_failures,
    ROUND(100.0 * SUM(failure) / COUNT(*), 2) AS failure_rate_pct
FROM telemetry
GROUP BY server_type
ORDER BY failure_rate_pct DESC;


-- 2. Failure Rate by Region
SELECT 
    region,
    COUNT(*) AS total_observations,
    SUM(failure) AS total_failures,
    ROUND(100.0 * SUM(failure) / COUNT(*), 2) AS failure_rate_pct
FROM telemetry
GROUP BY region
ORDER BY failure_rate_pct DESC;


-- 3. Telemetry Comparison: Normal vs Failure State
SELECT 
    failure,
    COUNT(*) AS obs_count,
    ROUND(AVG(cpu_usage), 2) AS avg_cpu,
    ROUND(AVG(memory_usage), 2) AS avg_memory,
    ROUND(AVG(network_latency), 2) AS avg_latency_ms,
    ROUND(AVG(packet_loss), 2) AS avg_packet_loss_pct,
    ROUND(AVG(error_rate), 2) AS avg_error_rate,
    ROUND(AVG(temperature), 2) AS avg_temp_celsius
FROM telemetry
GROUP BY failure;


-- 4. Top 10 High-Risk Servers (Highest Cumulative Failures & Avg Stress)
SELECT 
    server_id,
    server_type,
    region,
    COUNT(*) AS total_hours,
    SUM(failure) AS failure_count,
    ROUND(AVG(cpu_usage), 2) AS avg_cpu_usage,
    ROUND(AVG(memory_usage), 2) AS avg_memory_usage,
    ROUND(AVG(error_rate), 2) AS avg_error_rate
FROM telemetry
GROUP BY server_id, server_type, region
ORDER BY failure_count DESC, avg_cpu_usage DESC
LIMIT 10;


-- 5. Temporal Failure Trends (Hourly Aggregations across Dataset)
SELECT 
    SUBSTR(timestamp, 1, 13) || ':00:00' AS hour_window,
    COUNT(*) AS active_servers,
    SUM(failure) AS hourly_failures,
    ROUND(AVG(cpu_usage), 2) AS avg_system_cpu,
    ROUND(AVG(network_latency), 2) AS avg_system_latency
FROM telemetry
GROUP BY hour_window
ORDER BY hour_window ASC;


-- 6. Pre-Failure Stress Analysis (Telemetry 1 to 3 hours prior to failure)
SELECT 
    t.server_id,
    t.timestamp AS failure_time,
    t.server_type,
    t.cpu_usage AS failure_cpu,
    t.memory_usage AS failure_memory,
    t.network_latency AS failure_latency,
    t.error_rate AS failure_error_rate
FROM telemetry t
WHERE t.failure = 1
ORDER BY t.timestamp DESC
LIMIT 20;
