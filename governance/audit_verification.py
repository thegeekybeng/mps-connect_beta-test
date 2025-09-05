"""Audit verification and integrity checking for MPS Connect governance."""

import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

from sqlalchemy.orm import Session
from sqlalchemy import text

from database.connection import get_db

logger = logging.getLogger(__name__)


@dataclass
class AuditIntegrityResult:
    """Audit integrity verification result."""

    is_valid: bool
    total_records: int
    valid_records: int
    invalid_records: int
    broken_links: List[Dict[str, Any]]
    chain_integrity_percentage: float
    first_record_timestamp: Optional[datetime]
    last_record_timestamp: Optional[datetime]


class AuditVerifier:
    """Verifies audit log integrity and chain validation.

    What Audit Verification provides:
    - Immutable audit trail validation
    - Chain of custody verification
    - Tamper detection and prevention
    - Forensic evidence integrity
    - Compliance audit support
    """

    def __init__(self, db: Session):
        self.db = db

    def verify_audit_chain(self) -> AuditIntegrityResult:
        """Verify the complete audit chain integrity.

        What audit chain verification provides:
        - Complete chain validation
        - Tamper detection
        - Data integrity assurance
        - Forensic evidence validation
        - Compliance verification
        """
        try:
            # Get audit chain verification results
            verification_results = self.db.execute(
                text(
                    """
                SELECT 
                    block_number,
                    table_name,
                    record_id,
                    action,
                    created_at,
                    hash_chain,
                    previous_hash,
                    is_valid,
                    expected_hash
                FROM verify_audit_chain_integrity()
                ORDER BY block_number
            """
                )
            ).fetchall()

            total_records = len(verification_results)
            valid_records = sum(1 for r in verification_results if r.is_valid)
            invalid_records = total_records - valid_records

            # Get broken links details
            broken_links = [
                {
                    "block_number": r.block_number,
                    "table_name": r.table_name,
                    "record_id": str(r.record_id),
                    "action": r.action,
                    "created_at": r.created_at.isoformat(),
                    "current_hash": r.hash_chain,
                    "expected_hash": r.expected_hash,
                    "previous_hash": r.previous_hash,
                }
                for r in verification_results
                if not r.is_valid
            ]

            # Calculate integrity percentage
            chain_integrity_percentage = (
                (valid_records / total_records * 100) if total_records > 0 else 0
            )

            # Get timestamp range
            first_record_timestamp = (
                verification_results[0].created_at if verification_results else None
            )
            last_record_timestamp = (
                verification_results[-1].created_at if verification_results else None
            )

            return AuditIntegrityResult(
                is_valid=invalid_records == 0,
                total_records=total_records,
                valid_records=valid_records,
                invalid_records=invalid_records,
                broken_links=broken_links,
                chain_integrity_percentage=chain_integrity_percentage,
                first_record_timestamp=first_record_timestamp,
                last_record_timestamp=last_record_timestamp,
            )

        except Exception as e:
            logger.error("Audit chain verification failed: %s", e)
            return AuditIntegrityResult(
                is_valid=False,
                total_records=0,
                valid_records=0,
                invalid_records=0,
                broken_links=[{"error": str(e)}],
                chain_integrity_percentage=0.0,
                first_record_timestamp=None,
                last_record_timestamp=None,
            )

    def check_audit_integrity(self) -> Dict[str, Any]:
        """Check audit integrity and return summary.

        What audit integrity checking provides:
        - Quick integrity assessment
        - Chain validation summary
        - Compliance status check
        - Risk assessment
        """
        try:
            # Get audit chain summary
            summary = self.db.execute(
                text(
                    """
                SELECT * FROM get_audit_chain_summary()
            """
                )
            ).fetchone()

            # Get detailed verification
            integrity_result = self.verify_audit_chain()

            # Determine overall status
            if integrity_result.is_valid:
                status = "INTACT"
                risk_level = "LOW"
            elif integrity_result.chain_integrity_percentage >= 95:
                status = "MINOR_CORRUPTION"
                risk_level = "MEDIUM"
            else:
                status = "MAJOR_CORRUPTION"
                risk_level = "HIGH"

            return {
                "status": status,
                "risk_level": risk_level,
                "total_records": summary.total_records,
                "first_record_timestamp": (
                    summary.first_record_timestamp.isoformat()
                    if summary.first_record_timestamp
                    else None
                ),
                "last_record_timestamp": (
                    summary.last_record_timestamp.isoformat()
                    if summary.last_record_timestamp
                    else None
                ),
                "chain_integrity_status": summary.chain_integrity_status,
                "broken_links_count": summary.broken_links_count,
                "integrity_percentage": integrity_result.chain_integrity_percentage,
                "broken_links": integrity_result.broken_links[:10],  # Limit to first 10
                "verification_timestamp": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error("Audit integrity check failed: %s", e)
            return {
                "status": "ERROR",
                "risk_level": "CRITICAL",
                "error": str(e),
                "verification_timestamp": datetime.utcnow().isoformat(),
            }

    def get_audit_summary(
        self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get audit summary for specified period.

        What audit summary provides:
        - Period-based audit analysis
        - Activity patterns and trends
        - Compliance metrics
        - Risk indicators
        """
        try:
            # Build date filter
            date_filter = ""
            params = {}
            if start_date:
                date_filter += " AND created_at >= :start_date"
                params["start_date"] = start_date
            if end_date:
                date_filter += " AND created_at <= :end_date"
                params["end_date"] = end_date

            # Get audit summary
            summary_query = f"""
                SELECT 
                    COUNT(*) as total_records,
                    COUNT(CASE WHEN action = 'INSERT' THEN 1 END) as insert_count,
                    COUNT(CASE WHEN action = 'UPDATE' THEN 1 END) as update_count,
                    COUNT(CASE WHEN action = 'DELETE' THEN 1 END) as delete_count,
                    COUNT(DISTINCT table_name) as tables_affected,
                    COUNT(DISTINCT user_id) as users_involved,
                    MIN(created_at) as first_activity,
                    MAX(created_at) as last_activity
                FROM immutable_audit_logs
                WHERE 1=1 {date_filter}
            """

            summary = self.db.execute(text(summary_query), params).fetchone()

            # Get activity by table
            table_activity_query = f"""
                SELECT 
                    table_name,
                    COUNT(*) as record_count,
                    COUNT(CASE WHEN action = 'INSERT' THEN 1 END) as inserts,
                    COUNT(CASE WHEN action = 'UPDATE' THEN 1 END) as updates,
                    COUNT(CASE WHEN action = 'DELETE' THEN 1 END) as deletes
                FROM immutable_audit_logs
                WHERE 1=1 {date_filter}
                GROUP BY table_name
                ORDER BY record_count DESC
            """

            table_activities = self.db.execute(
                text(table_activity_query), params
            ).fetchall()

            # Get user activity
            user_activity_query = f"""
                SELECT 
                    u.email as user_email,
                    u.name as user_name,
                    u.role as user_role,
                    COUNT(*) as activity_count
                FROM immutable_audit_logs ial
                LEFT JOIN users u ON ial.user_id = u.id
                WHERE 1=1 {date_filter}
                GROUP BY u.id, u.email, u.name, u.role
                ORDER BY activity_count DESC
                LIMIT 20
            """

            user_activities = self.db.execute(
                text(user_activity_query), params
            ).fetchall()

            return {
                "period": {
                    "start_date": start_date.isoformat() if start_date else None,
                    "end_date": end_date.isoformat() if end_date else None,
                },
                "summary": {
                    "total_records": summary.total_records,
                    "insert_count": summary.insert_count,
                    "update_count": summary.update_count,
                    "delete_count": summary.delete_count,
                    "tables_affected": summary.tables_affected,
                    "users_involved": summary.users_involved,
                    "first_activity": (
                        summary.first_activity.isoformat()
                        if summary.first_activity
                        else None
                    ),
                    "last_activity": (
                        summary.last_activity.isoformat()
                        if summary.last_activity
                        else None
                    ),
                },
                "table_activities": [
                    {
                        "table_name": t.table_name,
                        "record_count": t.record_count,
                        "inserts": t.inserts,
                        "updates": t.updates,
                        "deletes": t.deletes,
                    }
                    for t in table_activities
                ],
                "user_activities": [
                    {
                        "user_email": u.user_email,
                        "user_name": u.user_name,
                        "user_role": u.user_role,
                        "activity_count": u.activity_count,
                    }
                    for u in user_activities
                ],
                "generated_at": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error("Audit summary generation failed: %s", e)
            return {"error": str(e), "generated_at": datetime.utcnow().isoformat()}

    def validate_audit_records(self, record_ids: List[str]) -> Dict[str, Any]:
        """Validate specific audit records.

        What audit record validation provides:
        - Individual record verification
        - Chain position validation
        - Integrity confirmation
        - Forensic evidence validation
        """
        try:
            if not record_ids:
                return {
                    "validated_records": [],
                    "summary": {"total": 0, "valid": 0, "invalid": 0},
                }

            # Convert string IDs to UUIDs for query
            record_id_list = "', '".join(record_ids)

            # Get specific records
            records_query = f"""
                SELECT 
                    block_number,
                    table_name,
                    record_id,
                    action,
                    created_at,
                    hash_chain,
                    previous_hash,
                    is_valid,
                    expected_hash
                FROM verify_audit_chain_integrity()
                WHERE record_id::text IN ('{record_id_list}')
                ORDER BY block_number
            """

            records = self.db.execute(text(records_query)).fetchall()

            validated_records = []
            valid_count = 0
            invalid_count = 0

            for record in records:
                is_valid = record.is_valid
                if is_valid:
                    valid_count += 1
                else:
                    invalid_count += 1

                validated_records.append(
                    {
                        "block_number": record.block_number,
                        "table_name": record.table_name,
                        "record_id": str(record.record_id),
                        "action": record.action,
                        "created_at": record.created_at.isoformat(),
                        "is_valid": is_valid,
                        "current_hash": record.hash_chain,
                        "expected_hash": record.expected_hash,
                        "previous_hash": record.previous_hash,
                    }
                )

            return {
                "validated_records": validated_records,
                "summary": {
                    "total": len(validated_records),
                    "valid": valid_count,
                    "invalid": invalid_count,
                },
                "validation_timestamp": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error("Audit record validation failed: %s", e)
            return {
                "error": str(e),
                "validation_timestamp": datetime.utcnow().isoformat(),
            }

    def get_chain_verification_report(self) -> Dict[str, Any]:
        """Generate comprehensive chain verification report.

        What chain verification report provides:
        - Complete audit chain analysis
        - Integrity assessment
        - Risk evaluation
        - Compliance status
        - Recommendations
        """
        try:
            # Get integrity result
            integrity_result = self.verify_audit_chain()

            # Get chain summary
            chain_summary = self.db.execute(
                text(
                    """
                SELECT * FROM get_audit_chain_summary()
            """
                )
            ).fetchone()

            # Determine recommendations
            recommendations = []
            if not integrity_result.is_valid:
                if integrity_result.chain_integrity_percentage >= 95:
                    recommendations.append(
                        "Minor corruption detected - monitor closely"
                    )
                    recommendations.append(
                        "Consider backup restoration for affected records"
                    )
                else:
                    recommendations.append(
                        "Major corruption detected - immediate action required"
                    )
                    recommendations.append("Consider full audit chain reconstruction")
                    recommendations.append("Review system security and access controls")
            else:
                recommendations.append(
                    "Audit chain is intact - maintain current practices"
                )
                recommendations.append("Continue regular integrity monitoring")

            # Calculate risk score
            if integrity_result.is_valid:
                risk_score = 0
            elif integrity_result.chain_integrity_percentage >= 95:
                risk_score = 25
            elif integrity_result.chain_integrity_percentage >= 80:
                risk_score = 50
            else:
                risk_score = 100

            return {
                "report_id": f"chain_verification_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                "generated_at": datetime.utcnow().isoformat(),
                "integrity_status": {
                    "is_valid": integrity_result.is_valid,
                    "total_records": integrity_result.total_records,
                    "valid_records": integrity_result.valid_records,
                    "invalid_records": integrity_result.invalid_records,
                    "integrity_percentage": integrity_result.chain_integrity_percentage,
                    "risk_score": risk_score,
                },
                "chain_summary": {
                    "total_records": chain_summary.total_records,
                    "first_record_timestamp": (
                        chain_summary.first_record_timestamp.isoformat()
                        if chain_summary.first_record_timestamp
                        else None
                    ),
                    "last_record_timestamp": (
                        chain_summary.last_record_timestamp.isoformat()
                        if chain_summary.last_record_timestamp
                        else None
                    ),
                    "chain_integrity_status": chain_summary.chain_integrity_status,
                    "broken_links_count": chain_summary.broken_links_count,
                },
                "broken_links": integrity_result.broken_links[:20],  # Limit to first 20
                "recommendations": recommendations,
                "compliance_status": (
                    "COMPLIANT" if integrity_result.is_valid else "NON_COMPLIANT"
                ),
            }

        except Exception as e:
            logger.error("Chain verification report generation failed: %s", e)
            return {"error": str(e), "generated_at": datetime.utcnow().isoformat()}


def verify_audit_chain() -> AuditIntegrityResult:
    """Verify audit chain integrity."""
    db = next(get_db())
    verifier = AuditVerifier(db)
    return verifier.verify_audit_chain()


def check_audit_integrity() -> Dict[str, Any]:
    """Check audit integrity."""
    db = next(get_db())
    verifier = AuditVerifier(db)
    return verifier.check_audit_integrity()


def get_audit_summary(
    start_date: Optional[datetime] = None, end_date: Optional[datetime] = None
) -> Dict[str, Any]:
    """Get audit summary."""
    db = next(get_db())
    verifier = AuditVerifier(db)
    return verifier.get_audit_summary(start_date, end_date)


def validate_audit_records(record_ids: List[str]) -> Dict[str, Any]:
    """Validate audit records."""
    db = next(get_db())
    verifier = AuditVerifier(db)
    return verifier.validate_audit_records(record_ids)
