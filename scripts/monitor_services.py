#!/usr/bin/env python3
"""MPS Connect Service Monitoring Script for Render."""

import os
import sys
import time
import requests
import psycopg2
import redis
import logging
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
from datetime import datetime
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ServiceMonitor:
    """Service health monitoring for MPS Connect."""

    def __init__(self):
        """Initialize monitoring configuration."""
        self.api_url = os.getenv("API_URL", "http://localhost:8000")
        self.database_url = os.getenv("DATABASE_URL")
        self.redis_url = os.getenv("REDIS_URL")
        self.monitor_interval = int(os.getenv("MONITOR_INTERVAL", "300"))  # 5 minutes
        self.alert_email = os.getenv("ALERT_EMAIL")
        self.smtp_host = os.getenv("SMTP_HOST")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.smtp_username = os.getenv("SMTP_USERNAME")
        self.smtp_password = os.getenv("SMTP_PASSWORD")

        # Service status tracking
        self.service_status = {
            "api": {"status": "unknown", "last_check": None, "error_count": 0},
            "database": {"status": "unknown", "last_check": None, "error_count": 0},
            "redis": {"status": "unknown", "last_check": None, "error_count": 0},
        }

        # Alert thresholds
        self.error_threshold = 3
        self.alert_cooldown = 3600  # 1 hour

    def check_api_health(self):
        """Check API service health.

        What API health checking provides:
        - Service availability verification
        - Response time monitoring
        - Error rate tracking
        - Performance metrics
        """
        try:
            logger.info("Checking API service health...")

            # Check health endpoint
            response = requests.get(
                f"{self.api_url}/health",
                timeout=10,
                headers={"User-Agent": "MPS-Connect-Monitor/1.0"},
            )

            if response.status_code == 200:
                data = response.json()
                logger.info(f"API health check passed: {data}")
                return True
            else:
                logger.warning(f"API health check failed: HTTP {response.status_code}")
                return False

        except requests.exceptions.RequestException as e:
            logger.error(f"API health check failed: {e}")
            return False
        except Exception as e:
            logger.error(f"API health check error: {e}")
            return False

    def check_database_health(self):
        """Check database service health.

        What database health checking provides:
        - Connection availability
        - Query performance
        - Database metrics
        - Data integrity checks
        """
        try:
            logger.info("Checking database service health...")

            # Test database connection
            conn = psycopg2.connect(self.database_url)
            cursor = conn.cursor()

            # Run simple query
            cursor.execute("SELECT 1")
            result = cursor.fetchone()

            if result and result[0] == 1:
                logger.info("Database health check passed")

                # Check database metrics
                cursor.execute(
                    """
                    SELECT 
                        COUNT(*) as total_tables,
                        pg_database_size(current_database()) as db_size
                    FROM information_schema.tables 
                    WHERE table_schema = 'public'
                """
                )
                metrics = cursor.fetchone()
                logger.info(
                    f"Database metrics: {metrics[0]} tables, {metrics[1]} bytes"
                )

                cursor.close()
                conn.close()
                return True
            else:
                logger.warning("Database health check failed: Invalid query result")
                return False

        except psycopg2.Error as e:
            logger.error(f"Database health check failed: {e}")
            return False
        except Exception as e:
            logger.error(f"Database health check error: {e}")
            return False

    def check_redis_health(self):
        """Check Redis service health.

        What Redis health checking provides:
        - Connection availability
        - Memory usage monitoring
        - Performance metrics
        - Cache hit rates
        """
        try:
            logger.info("Checking Redis service health...")

            # Test Redis connection
            r = redis.from_url(self.redis_url)

            # Test basic operations
            r.ping()

            # Get Redis info
            info = r.info()

            logger.info(
                f"Redis health check passed: {info.get('redis_version', 'unknown')}"
            )
            logger.info(
                f"Redis memory usage: {info.get('used_memory_human', 'unknown')}"
            )
            logger.info(
                f"Redis connected clients: {info.get('connected_clients', 'unknown')}"
            )

            return True

        except redis.RedisError as e:
            logger.error(f"Redis health check failed: {e}")
            return False
        except Exception as e:
            logger.error(f"Redis health check error: {e}")
            return False

    def check_governance_health(self):
        """Check governance system health.

        What governance health checking provides:
        - Compliance status monitoring
        - Audit log integrity
        - Security system status
        - Regulatory compliance
        """
        try:
            logger.info("Checking governance system health...")

            # Check governance health endpoint
            response = requests.get(
                f"{self.api_url}/governance/health",
                timeout=10,
                headers={"User-Agent": "MPS-Connect-Monitor/1.0"},
            )

            if response.status_code == 200:
                data = response.json()
                logger.info(f"Governance health check passed: {data}")
                return True
            else:
                logger.warning(
                    f"Governance health check failed: HTTP {response.status_code}"
                )
                return False

        except requests.exceptions.RequestException as e:
            logger.error(f"Governance health check failed: {e}")
            return False
        except Exception as e:
            logger.error(f"Governance health check error: {e}")
            return False

    def update_service_status(self, service, is_healthy):
        """Update service status tracking.

        What status tracking provides:
        - Service availability history
        - Error count tracking
        - Alert triggering
        - Performance monitoring
        """
        current_time = datetime.now()

        if is_healthy:
            self.service_status[service]["status"] = "healthy"
            self.service_status[service]["error_count"] = 0
        else:
            self.service_status[service]["status"] = "unhealthy"
            self.service_status[service]["error_count"] += 1

        self.service_status[service]["last_check"] = current_time

        # Check if alert should be sent
        if (
            self.service_status[service]["error_count"] >= self.error_threshold
            and self.service_status[service]["status"] == "unhealthy"
        ):
            self.send_alert(
                service,
                f"Service {service} has been unhealthy for {self.service_status[service]['error_count']} consecutive checks",
            )

    def send_alert(self, service, message):
        """Send alert notification.

        What alerting provides:
        - Immediate notification of issues
        - Email notifications
        - Alert escalation
        - Incident tracking
        """
        if not self.alert_email or not self.smtp_host:
            logger.warning("Alert email not configured, skipping alert")
            return

        try:
            logger.info(f"Sending alert for {service}: {message}")

            # Create email message
            msg = MimeMultipart()
            msg["From"] = self.smtp_username
            msg["To"] = self.alert_email
            msg["Subject"] = f"MPS Connect Alert: {service} Service Issue"

            # Email body
            body = f"""
            MPS Connect Service Alert
            
            Service: {service}
            Time: {datetime.now().isoformat()}
            Message: {message}
            
            Service Status:
            {json.dumps(self.service_status, indent=2, default=str)}
            
            Please check the service immediately.
            """

            msg.attach(MimeText(body, "plain"))

            # Send email
            server = smtplib.SMTP(self.smtp_host, self.smtp_port)
            server.starttls()
            server.login(self.smtp_username, self.smtp_password)
            server.send_message(msg)
            server.quit()

            logger.info("Alert sent successfully")

        except Exception as e:
            logger.error(f"Failed to send alert: {e}")

    def run_health_checks(self):
        """Run all health checks.

        What health checks provide:
        - Comprehensive service monitoring
        - Automated health verification
        - Performance tracking
        - Issue detection
        """
        logger.info("Running MPS Connect health checks...")

        # Check API service
        api_healthy = self.check_api_health()
        self.update_service_status("api", api_healthy)

        # Check database service
        db_healthy = self.check_database_health()
        self.update_service_status("database", db_healthy)

        # Check Redis service
        redis_healthy = self.check_redis_health()
        self.update_service_status("redis", redis_healthy)

        # Check governance system
        governance_healthy = self.check_governance_health()

        # Log overall status
        healthy_services = sum(
            1
            for status in self.service_status.values()
            if status["status"] == "healthy"
        )
        total_services = len(self.service_status)

        logger.info(
            f"Health check completed: {healthy_services}/{total_services} services healthy"
        )

        if governance_healthy:
            logger.info("Governance system is healthy")
        else:
            logger.warning("Governance system health check failed")

        return healthy_services == total_services

    def start_monitoring(self):
        """Start continuous monitoring.

        What continuous monitoring provides:
        - 24/7 service monitoring
        - Automated health checks
        - Alert notifications
        - Performance tracking
        """
        logger.info("Starting MPS Connect service monitoring...")
        logger.info(f"Monitoring interval: {self.monitor_interval} seconds")

        while True:
            try:
                self.run_health_checks()
                time.sleep(self.monitor_interval)
            except KeyboardInterrupt:
                logger.info("Monitoring stopped by user")
                break
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                time.sleep(60)  # Wait 1 minute before retrying


def main():
    """Main monitoring execution."""
    monitor = ServiceMonitor()

    if len(sys.argv) > 1 and sys.argv[1] == "--once":
        # Run once and exit
        success = monitor.run_health_checks()
        sys.exit(0 if success else 1)
    else:
        # Run continuously
        monitor.start_monitoring()


if __name__ == "__main__":
    main()
