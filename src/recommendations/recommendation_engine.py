"""
Infrastructure Intelligence - Infrastructure Action Recommendation Engine
Generates explicit, rule-based operational recommendations tied to system telemetry
triggers, SHAP root causes, and risk classifications.
"""

from typing import Dict, Any, List


class RecommendationEngine:
    """Rule-based engine mapping system telemetry triggers to actionable IT ops recommendations."""

    @staticmethod
    def generate_recommendations(
        telemetry: Dict[str, Any],
        risk_level: str,
        shap_top_positive: List[Dict[str, Any]] = None
    ) -> List[Dict[str, str]]:
        """Generate structured operational action items based on telemetry triggers."""
        actions = []

        cpu = float(telemetry.get('cpu_usage', 0.0))
        memory = float(telemetry.get('memory_usage', 0.0))
        disk = float(telemetry.get('disk_usage', 0.0))
        latency = float(telemetry.get('network_latency', 0.0))
        packet_loss = float(telemetry.get('packet_loss', 0.0))
        error_rate = float(telemetry.get('error_rate', 0.0))
        temperature = float(telemetry.get('temperature', 0.0))
        prev_failures = int(telemetry.get('previous_failures', 0))

        # Rule 1: High CPU Saturation
        if cpu >= 85.0:
            actions.append({
                'category': 'CPU Saturation',
                'severity': 'HIGH' if cpu >= 92.0 else 'MEDIUM',
                'trigger': f'CPU Usage at {cpu:.1f}%',
                'action': 'Inspect top CPU-consuming processes (top/htop), analyze thread contention, and execute load balancing or horizontal auto-scaling.'
            })

        # Rule 2: Memory Exhaustion / Leak
        if memory >= 88.0:
            actions.append({
                'category': 'Memory Contention',
                'severity': 'HIGH' if memory >= 94.0 else 'MEDIUM',
                'trigger': f'Memory Usage at {memory:.1f}%',
                'action': 'Investigate potential memory leaks, clear OS page caches, inspect JVM heap/database buffer pools, and recycle bloated service workers.'
            })

        # Rule 3: Network Latency & Packet Loss Coupling
        if latency >= 120.0 or packet_loss >= 3.0:
            actions.append({
                'category': 'Network Degradation',
                'severity': 'HIGH' if latency >= 180.0 else 'MEDIUM',
                'trigger': f'Latency {latency:.1f}ms, Packet Loss {packet_loss:.1f}%',
                'action': 'Inspect network interface card (NIC) counters, check switch port buffer drops, verify routing table hops, and reroute network traffic.'
            })

        # Rule 4: High Application Error Rate
        if error_rate >= 25.0:
            actions.append({
                'category': 'Service Failure / Error Spike',
                'severity': 'HIGH' if error_rate >= 60.0 else 'MEDIUM',
                'trigger': f'Error Rate at {error_rate:.1f} errors/sec',
                'action': 'Inspect application centralized log streams (ELK/Splunk), check downstream database connection pools, and roll back recent deployment if necessary.'
            })

        # Rule 5: Thermal Overheating Alert
        if temperature >= 82.0:
            actions.append({
                'category': 'Thermal Heating Risk',
                'severity': 'CRITICAL' if temperature >= 90.0 else 'HIGH',
                'trigger': f'Chassis Temp at {temperature:.1f}°C',
                'action': 'Check server rack airflow, verify HVAC datacenter cooling performance, inspect chassis fan RPMs, and throttle workload intensity.'
            })

        # Rule 6: High Disk Usage
        if disk >= 90.0:
            actions.append({
                'category': 'Storage Exhaustion',
                'severity': 'MEDIUM',
                'trigger': f'Disk Usage at {disk:.1f}%',
                'action': 'Purge temporary log archives, run automated disk cleanup scripts, and provision additional storage volume space.'
            })

        # Rule 7: Repeated Past Failures
        if prev_failures >= 2:
            actions.append({
                'category': 'Recurrent Failure History',
                'severity': 'HIGH',
                'trigger': f'Server experienced {prev_failures} past failure(s)',
                'action': 'Schedule comprehensive physical hardware diagnostics, check RAM ECC memory error logs, and consider server hardware replacement.'
            })

        # Rule 8: High Risk Escalation Rule
        if risk_level == 'HIGH RISK' and len(actions) >= 2:
            actions.insert(0, {
                'category': 'IMMEDIATE L3 ESCALATION',
                'severity': 'CRITICAL',
                'trigger': f'Multiple compound telemetry risk factors under HIGH RISK status',
                'action': 'Trigger automated PagerDuty alert, failover active workloads to standby secondary region, and initiate emergency incident response protocol.'
            })

        # Default Healthy Recommendation
        if not actions:
            actions.append({
                'category': 'Normal Maintenance',
                'severity': 'INFO',
                'trigger': 'All system metrics operating within normal operating parameters',
                'action': 'Maintain routine telemetry monitoring. No immediate manual intervention required.'
            })

        return actions


if __name__ == "__main__":
    sample_telemetry = {
        'cpu_usage': 94.0,
        'memory_usage': 92.5,
        'network_latency': 160.0,
        'packet_loss': 4.0,
        'error_rate': 45.0,
        'temperature': 88.0,
        'previous_failures': 2
    }
    recs = RecommendationEngine.generate_recommendations(sample_telemetry, risk_level='HIGH RISK')
    print("Generated Recommendations:")
    for r in recs:
        print(f"[{r['severity']}] {r['category']}: {r['action']}")
