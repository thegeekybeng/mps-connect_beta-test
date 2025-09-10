"""Resource monitoring for transparency in MPS Connect AI system."""

import psutil
import time
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ResourceSnapshot:
    """Data class for resource snapshots."""

    timestamp: datetime
    cpu_percent: float
    memory_usage_mb: float
    memory_percent: float
    disk_usage_mb: float
    disk_percent: float
    network_sent_mb: float
    network_recv_mb: float


class ResourceMonitor:
    """Monitors system resources for transparency."""

    def __init__(self):
        self.active_monitoring = {}
        self.resource_history = {}
        self.baseline_resources = self._get_baseline_resources()

    def start_monitoring(self, session_id: str) -> None:
        """
        Start monitoring resources for a session.

        Args:
            session_id: Tracking session ID
        """
        try:
            # Record baseline resources
            baseline = self._get_current_resources()

            self.active_monitoring[session_id] = {
                "start_time": time.time(),
                "baseline": baseline,
                "snapshots": [],
                "peak_cpu": baseline.cpu_percent,
                "peak_memory": baseline.memory_usage_mb,
                "peak_disk": baseline.disk_usage_mb,
            }

            logger.info(f"Started resource monitoring for session {session_id}")

        except Exception as e:
            logger.error(f"Error starting resource monitoring: {str(e)}")

    def take_snapshot(self, session_id: str) -> Optional[ResourceSnapshot]:
        """
        Take a resource snapshot for a session.

        Args:
            session_id: Tracking session ID

        Returns:
            Resource snapshot or None if session not found
        """
        try:
            if session_id not in self.active_monitoring:
                logger.warning(
                    f"Session {session_id} not found for resource monitoring"
                )
                return None

            # Get current resources
            snapshot = self._get_current_resources()

            # Update monitoring data
            monitoring_data = self.active_monitoring[session_id]
            monitoring_data["snapshots"].append(snapshot)

            # Update peak values
            monitoring_data["peak_cpu"] = max(
                monitoring_data["peak_cpu"], snapshot.cpu_percent
            )
            monitoring_data["peak_memory"] = max(
                monitoring_data["peak_memory"], snapshot.memory_usage_mb
            )
            monitoring_data["peak_disk"] = max(
                monitoring_data["peak_disk"], snapshot.disk_usage_mb
            )

            logger.debug(f"Took resource snapshot for session {session_id}")
            return snapshot

        except Exception as e:
            logger.error(f"Error taking resource snapshot: {str(e)}")
            return None

    def stop_monitoring(self, session_id: str) -> Dict[str, Any]:
        """
        Stop monitoring and return resource usage summary.

        Args:
            session_id: Tracking session ID

        Returns:
            Resource usage summary
        """
        try:
            if session_id not in self.active_monitoring:
                logger.warning(
                    f"Session {session_id} not found for resource monitoring"
                )
                return {"error": "Session not found"}

            monitoring_data = self.active_monitoring[session_id]
            end_time = time.time()
            duration = end_time - monitoring_data["start_time"]

            # Take final snapshot
            final_snapshot = self._get_current_resources()
            monitoring_data["snapshots"].append(final_snapshot)

            # Calculate resource usage summary
            summary = self._calculate_resource_summary(monitoring_data, duration)

            # Store in history
            self.resource_history[session_id] = {
                "monitoring_data": monitoring_data,
                "summary": summary,
                "duration_seconds": duration,
            }

            # Clean up active monitoring
            del self.active_monitoring[session_id]

            logger.info(f"Stopped resource monitoring for session {session_id}")
            return summary

        except Exception as e:
            logger.error(f"Error stopping resource monitoring: {str(e)}")
            return {"error": str(e)}

    def _get_current_resources(self) -> ResourceSnapshot:
        """Get current system resource usage."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)

            # Memory usage
            memory = psutil.virtual_memory()
            memory_usage_mb = memory.used / (1024 * 1024)  # Convert to MB
            memory_percent = memory.percent

            # Disk usage
            disk = psutil.disk_usage("/")
            disk_usage_mb = disk.used / (1024 * 1024)  # Convert to MB
            disk_percent = (disk.used / disk.total) * 100

            # Network usage
            network = psutil.net_io_counters()
            network_sent_mb = network.bytes_sent / (1024 * 1024)  # Convert to MB
            network_recv_mb = network.bytes_recv / (1024 * 1024)  # Convert to MB

            return ResourceSnapshot(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_usage_mb=memory_usage_mb,
                memory_percent=memory_percent,
                disk_usage_mb=disk_usage_mb,
                disk_percent=disk_percent,
                network_sent_mb=network_sent_mb,
                network_recv_mb=network_recv_mb,
            )

        except Exception as e:
            logger.error(f"Error getting current resources: {str(e)}")
            # Return zero values as fallback
            return ResourceSnapshot(
                timestamp=datetime.now(),
                cpu_percent=0.0,
                memory_usage_mb=0.0,
                memory_percent=0.0,
                disk_usage_mb=0.0,
                disk_percent=0.0,
                network_sent_mb=0.0,
                network_recv_mb=0.0,
            )

    def _get_baseline_resources(self) -> ResourceSnapshot:
        """Get baseline resource usage."""
        try:
            # Take multiple samples to get a stable baseline
            samples = []
            for _ in range(3):
                samples.append(self._get_current_resources())
                time.sleep(0.1)

            # Calculate average
            avg_cpu = sum(s.cpu_percent for s in samples) / len(samples)
            avg_memory = sum(s.memory_usage_mb for s in samples) / len(samples)
            avg_memory_percent = sum(s.memory_percent for s in samples) / len(samples)
            avg_disk = sum(s.disk_usage_mb for s in samples) / len(samples)
            avg_disk_percent = sum(s.disk_percent for s in samples) / len(samples)
            avg_network_sent = sum(s.network_sent_mb for s in samples) / len(samples)
            avg_network_recv = sum(s.network_recv_mb for s in samples) / len(samples)

            return ResourceSnapshot(
                timestamp=datetime.now(),
                cpu_percent=avg_cpu,
                memory_usage_mb=avg_memory,
                memory_percent=avg_memory_percent,
                disk_usage_mb=avg_disk,
                disk_percent=avg_disk_percent,
                network_sent_mb=avg_network_sent,
                network_recv_mb=avg_network_recv,
            )

        except Exception as e:
            logger.error(f"Error getting baseline resources: {str(e)}")
            return self._get_current_resources()

    def _calculate_resource_summary(
        self, monitoring_data: Dict[str, Any], duration: float
    ) -> Dict[str, Any]:
        """Calculate resource usage summary."""
        try:
            snapshots = monitoring_data["snapshots"]
            baseline = monitoring_data["baseline"]

            if not snapshots:
                return {"error": "No snapshots available"}

            # Calculate averages
            avg_cpu = sum(s.cpu_percent for s in snapshots) / len(snapshots)
            avg_memory = sum(s.memory_usage_mb for s in snapshots) / len(snapshots)
            avg_memory_percent = sum(s.memory_percent for s in snapshots) / len(
                snapshots
            )
            avg_disk = sum(s.disk_usage_mb for s in snapshots) / len(snapshots)
            avg_disk_percent = sum(s.disk_percent for s in snapshots) / len(snapshots)

            # Calculate deltas from baseline
            cpu_delta = avg_cpu - baseline.cpu_percent
            memory_delta = avg_memory - baseline.memory_usage_mb
            disk_delta = avg_disk - baseline.disk_usage_mb

            # Calculate network usage
            if len(snapshots) > 1:
                network_sent_delta = (
                    snapshots[-1].network_sent_mb - snapshots[0].network_sent_mb
                )
                network_recv_delta = (
                    snapshots[-1].network_recv_mb - snapshots[0].network_recv_mb
                )
            else:
                network_sent_delta = 0
                network_recv_delta = 0

            # Calculate efficiency metrics
            cpu_efficiency = self._calculate_cpu_efficiency(avg_cpu, duration)
            memory_efficiency = self._calculate_memory_efficiency(avg_memory, duration)

            return {
                "duration_seconds": duration,
                "snapshot_count": len(snapshots),
                "cpu_usage": {
                    "average_percent": avg_cpu,
                    "peak_percent": monitoring_data["peak_cpu"],
                    "baseline_percent": baseline.cpu_percent,
                    "delta_percent": cpu_delta,
                    "efficiency_score": cpu_efficiency,
                },
                "memory_usage": {
                    "average_mb": avg_memory,
                    "peak_mb": monitoring_data["peak_memory"],
                    "baseline_mb": baseline.memory_usage_mb,
                    "delta_mb": memory_delta,
                    "average_percent": avg_memory_percent,
                    "efficiency_score": memory_efficiency,
                },
                "disk_usage": {
                    "average_mb": avg_disk,
                    "peak_mb": monitoring_data["peak_disk"],
                    "baseline_mb": baseline.disk_usage_mb,
                    "delta_mb": disk_delta,
                    "average_percent": avg_disk_percent,
                },
                "network_usage": {
                    "sent_mb": network_sent_delta,
                    "received_mb": network_recv_delta,
                    "total_mb": network_sent_delta + network_recv_delta,
                },
                "efficiency_summary": {
                    "overall_efficiency": (cpu_efficiency + memory_efficiency) / 2,
                    "resource_intensive": avg_cpu > 80 or avg_memory_percent > 80,
                    "efficient_processing": duration < 5.0 and avg_cpu < 50,
                },
            }

        except Exception as e:
            logger.error(f"Error calculating resource summary: {str(e)}")
            return {"error": str(e)}

    def _calculate_cpu_efficiency(self, avg_cpu: float, duration: float) -> float:
        """Calculate CPU efficiency score."""
        try:
            # Efficiency based on CPU usage and processing time
            # Lower CPU usage and shorter duration = higher efficiency
            cpu_score = max(0, 100 - avg_cpu)  # 100 - CPU percentage
            time_score = max(0, 100 - (duration * 10))  # Penalty for longer processing

            efficiency = (cpu_score + time_score) / 2
            return min(efficiency, 100)  # Cap at 100

        except Exception as e:
            logger.error(f"Error calculating CPU efficiency: {str(e)}")
            return 50.0  # Default middle score

    def _calculate_memory_efficiency(self, avg_memory: float, duration: float) -> float:
        """Calculate memory efficiency score."""
        try:
            # Efficiency based on memory usage and processing time
            # Lower memory usage and shorter duration = higher efficiency
            memory_score = max(
                0, 100 - (avg_memory / 100)
            )  # Penalty for high memory usage
            time_score = max(0, 100 - (duration * 10))  # Penalty for longer processing

            efficiency = (memory_score + time_score) / 2
            return min(efficiency, 100)  # Cap at 100

        except Exception as e:
            logger.error(f"Error calculating memory efficiency: {str(e)}")
            return 50.0  # Default middle score

    def get_resource_history(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get resource history for a session."""
        try:
            return self.resource_history.get(session_id)
        except Exception as e:
            logger.error(f"Error getting resource history: {str(e)}")
            return None

    def get_system_resources(self) -> Dict[str, Any]:
        """Get current system resource status."""
        try:
            current = self._get_current_resources()

            return {
                "timestamp": current.timestamp.isoformat(),
                "cpu_percent": current.cpu_percent,
                "memory_usage_mb": current.memory_usage_mb,
                "memory_percent": current.memory_percent,
                "disk_usage_mb": current.disk_usage_mb,
                "disk_percent": current.disk_percent,
                "network_sent_mb": current.network_sent_mb,
                "network_recv_mb": current.network_recv_mb,
                "baseline": {
                    "cpu_percent": self.baseline_resources.cpu_percent,
                    "memory_usage_mb": self.baseline_resources.memory_usage_mb,
                    "memory_percent": self.baseline_resources.memory_percent,
                },
            }

        except Exception as e:
            logger.error(f"Error getting system resources: {str(e)}")
            return {"error": str(e)}

    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get summary of all monitoring data."""
        try:
            active_sessions = list(self.active_monitoring.keys())
            historical_sessions = list(self.resource_history.keys())

            return {
                "active_monitoring_sessions": len(active_sessions),
                "historical_sessions": len(historical_sessions),
                "active_session_ids": active_sessions,
                "historical_session_ids": historical_sessions,
                "baseline_resources": {
                    "cpu_percent": self.baseline_resources.cpu_percent,
                    "memory_usage_mb": self.baseline_resources.memory_usage_mb,
                    "memory_percent": self.baseline_resources.memory_percent,
                },
            }

        except Exception as e:
            logger.error(f"Error getting monitoring summary: {str(e)}")
            return {"error": str(e)}

    def clear_session_data(self, session_id: str) -> None:
        """Clear monitoring data for a session."""
        try:
            if session_id in self.resource_history:
                del self.resource_history[session_id]
                logger.info(
                    f"Cleared resource monitoring data for session {session_id}"
                )

        except Exception as e:
            logger.error(f"Error clearing session data: {str(e)}")
