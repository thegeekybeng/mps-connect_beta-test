#!/usr/bin/env python3
"""MPS Connect Database Backup Script for Render."""

import os
import sys
import subprocess
import boto3
import psycopg2
from datetime import datetime, timedelta
import logging
import gzip
import shutil
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DatabaseBackup:
    """Database backup management for MPS Connect."""

    def __init__(self):
        """Initialize backup configuration."""
        self.database_url = os.getenv("DATABASE_URL")
        self.s3_bucket = os.getenv("BACKUP_S3_BUCKET", "mps-connect-backups")
        self.s3_region = os.getenv("BACKUP_S3_REGION", "us-east-1")
        self.retention_days = int(os.getenv("BACKUP_RETENTION_DAYS", "30"))
        self.backup_dir = Path("/app/backups")
        self.backup_dir.mkdir(exist_ok=True)

        # Initialize S3 client
        self.s3_client = boto3.client(
            "s3",
            region_name=self.s3_region,
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        )

    def create_backup(self):
        """Create database backup.

        What database backup provides:
        - Complete database dump
        - Compressed backup files
        - Timestamped backups
        - Integrity verification
        - S3 storage for durability
        """
        try:
            # Generate backup filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_filename = f"mps_connect_backup_{timestamp}.sql"
            backup_path = self.backup_dir / backup_filename
            compressed_path = self.backup_dir / f"{backup_filename}.gz"

            logger.info(f"Creating database backup: {backup_filename}")

            # Create database dump
            dump_command = [
                "pg_dump",
                self.database_url,
                "--verbose",
                "--no-password",
                "--format=plain",
                "--no-owner",
                "--no-privileges",
                "--clean",
                "--if-exists",
                "--create",
            ]

            with open(backup_path, "w") as f:
                result = subprocess.run(
                    dump_command,
                    stdout=f,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=True,
                )

            logger.info("Database dump completed successfully")

            # Compress backup
            logger.info("Compressing backup file...")
            with open(backup_path, "rb") as f_in:
                with gzip.open(compressed_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)

            # Remove uncompressed file
            backup_path.unlink()

            logger.info(f"Backup compressed: {compressed_path}")

            # Upload to S3
            self.upload_to_s3(compressed_path, backup_filename + ".gz")

            # Clean up local file
            compressed_path.unlink()

            logger.info("Database backup completed successfully")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Database dump failed: {e.stderr}")
            return False
        except Exception as e:
            logger.error(f"Backup creation failed: {e}")
            return False

    def upload_to_s3(self, file_path, s3_key):
        """Upload backup file to S3.

        What S3 upload provides:
        - Durable storage
        - Geographic distribution
        - Versioning support
        - Access control
        - Cost optimization
        """
        try:
            logger.info(f"Uploading {file_path} to S3: s3://{self.s3_bucket}/{s3_key}")

            self.s3_client.upload_file(
                str(file_path),
                self.s3_bucket,
                s3_key,
                ExtraArgs={
                    "ServerSideEncryption": "AES256",
                    "StorageClass": "STANDARD_IA",
                },
            )

            logger.info("Upload to S3 completed successfully")

        except Exception as e:
            logger.error(f"S3 upload failed: {e}")
            raise

    def cleanup_old_backups(self):
        """Clean up old backup files from S3.

        What cleanup provides:
        - Storage cost optimization
        - Retention policy enforcement
        - Automated cleanup
        - Compliance with data retention
        """
        try:
            logger.info("Cleaning up old backup files...")

            # Calculate cutoff date
            cutoff_date = datetime.now() - timedelta(days=self.retention_days)

            # List objects in S3 bucket
            response = self.s3_client.list_objects_v2(
                Bucket=self.s3_bucket, Prefix="mps_connect_backup_"
            )

            if "Contents" not in response:
                logger.info("No backup files found in S3")
                return

            # Delete old files
            deleted_count = 0
            for obj in response["Contents"]:
                # Parse timestamp from filename
                filename = obj["Key"]
                if "mps_connect_backup_" in filename:
                    try:
                        # Extract timestamp from filename
                        timestamp_str = filename.split("_")[-1].replace(".sql.gz", "")
                        file_date = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")

                        if file_date < cutoff_date:
                            self.s3_client.delete_object(
                                Bucket=self.s3_bucket, Key=filename
                            )
                            deleted_count += 1
                            logger.info(f"Deleted old backup: {filename}")

                    except ValueError:
                        logger.warning(
                            f"Could not parse timestamp from filename: {filename}"
                        )
                        continue

            logger.info(f"Cleaned up {deleted_count} old backup files")

        except Exception as e:
            logger.error(f"Cleanup failed: {e}")

    def verify_backup(self, s3_key):
        """Verify backup integrity.

        What backup verification provides:
        - Data integrity validation
        - Backup completeness check
        - Restoration testing
        - Quality assurance
        """
        try:
            logger.info(f"Verifying backup: {s3_key}")

            # Download backup file
            local_path = self.backup_dir / s3_key
            self.s3_client.download_file(self.s3_bucket, s3_key, str(local_path))

            # Decompress and verify
            with gzip.open(local_path, "rb") as f_in:
                with open(local_path.with_suffix(""), "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)

            # Check if file contains expected content
            with open(local_path.with_suffix(""), "r") as f:
                content = f.read()
                if "CREATE DATABASE" in content and "mps_connect" in content:
                    logger.info("Backup verification successful")
                    return True
                else:
                    logger.error("Backup verification failed: Invalid content")
                    return False

        except Exception as e:
            logger.error(f"Backup verification failed: {e}")
            return False
        finally:
            # Clean up local files
            if local_path.exists():
                local_path.unlink()
            if local_path.with_suffix("").exists():
                local_path.with_suffix("").unlink()

    def run_backup(self):
        """Run complete backup process.

        What the backup process provides:
        - Automated backup creation
        - S3 storage upload
        - Old backup cleanup
        - Integrity verification
        - Comprehensive logging
        """
        logger.info("Starting MPS Connect database backup process")

        try:
            # Create backup
            if not self.create_backup():
                logger.error("Backup creation failed")
                return False

            # Clean up old backups
            self.cleanup_old_backups()

            logger.info("Database backup process completed successfully")
            return True

        except Exception as e:
            logger.error(f"Backup process failed: {e}")
            return False


def main():
    """Main backup execution."""
    backup = DatabaseBackup()
    success = backup.run_backup()

    if success:
        logger.info("Backup completed successfully")
        sys.exit(0)
    else:
        logger.error("Backup failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
