"""Compliance management for MPS Connect governance."""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from enum import Enum
from dataclasses import dataclass

from sqlalchemy.orm import Session
from sqlalchemy import text

from database.connection import get_db
from database.models import User, Case, Letter, AuditLog, UserActivity, AccessLog

logger = logging.getLogger(__name__)


class ComplianceStatus(Enum):
    """Compliance status enumeration."""

    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    AT_RISK = "at_risk"
    UNKNOWN = "unknown"


@dataclass
class ComplianceReport:
    """Compliance report data structure."""

    report_id: str
    generated_at: datetime
    period_start: datetime
    period_end: datetime
    overall_status: ComplianceStatus
    compliance_score: float
    total_requirements: int
    met_requirements: int
    failed_requirements: int
    at_risk_requirements: int
    details: Dict[str, Any]
    recommendations: List[str]


class ComplianceManager:
    """Manages compliance requirements and reporting.

    What Compliance Management provides:
    - Regulatory requirement tracking
    - Automated compliance checking
    - Compliance reporting and documentation
    - Risk assessment and mitigation
    - Policy enforcement and monitoring
    """

    def __init__(self, db: Session):
        self.db = db
        self.requirements = self._load_compliance_requirements()

    def _load_compliance_requirements(self) -> Dict[str, Dict[str, Any]]:
        """Load compliance requirements configuration.

        What compliance requirements provide:
        - Government regulatory compliance
        - Data protection requirements
        - Security standards compliance
        - Audit trail requirements
        - Retention policy compliance
        """
        return {
            "audit_logging": {
                "name": "Audit Logging",
                "description": "All data changes must be logged with immutable audit trail",
                "severity": "critical",
                "check_function": self._check_audit_logging,
                "weight": 0.3,
            },
            "data_retention": {
                "name": "Data Retention",
                "description": "Data must be retained according to policy and cleaned up appropriately",
                "severity": "high",
                "check_function": self._check_data_retention,
                "weight": 0.2,
            },
            "user_authentication": {
                "name": "User Authentication",
                "description": "All users must be properly authenticated and authorized",
                "severity": "critical",
                "check_function": self._check_user_authentication,
                "weight": 0.25,
            },
            "data_encryption": {
                "name": "Data Encryption",
                "description": "Sensitive data must be encrypted at rest and in transit",
                "severity": "critical",
                "check_function": self._check_data_encryption,
                "weight": 0.15,
            },
            "access_control": {
                "name": "Access Control",
                "description": "Role-based access control must be properly implemented",
                "severity": "high",
                "check_function": self._check_access_control,
                "weight": 0.1,
            },
        }

    def check_compliance_status(
        self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None
    ) -> ComplianceReport:
        """Check overall compliance status.

        What compliance checking provides:
        - Automated compliance validation
        - Risk assessment and scoring
        - Detailed compliance reporting
        - Regulatory requirement verification
        - Policy adherence monitoring
        """
        if not start_date:
            start_date = datetime.utcnow() - timedelta(days=30)
        if not end_date:
            end_date = datetime.utcnow()

        report_id = f"compliance_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        # Check each requirement
        requirement_results = {}
        total_weight = 0
        weighted_score = 0

        for req_id, req_config in self.requirements.items():
            try:
                result = req_config["check_function"](start_date, end_date)
                requirement_results[req_id] = result

                # Calculate weighted score
                weight = req_config["weight"]
                total_weight += weight
                weighted_score += result["score"] * weight

            except Exception as e:
                logger.error("Failed to check requirement %s: %s", req_id, e)
                requirement_results[req_id] = {
                    "status": ComplianceStatus.UNKNOWN,
                    "score": 0.0,
                    "details": {"error": str(e)},
                    "recommendations": ["Fix requirement check function"],
                }

        # Calculate overall compliance score
        overall_score = weighted_score / total_weight if total_weight > 0 else 0.0

        # Determine overall status
        if overall_score >= 0.9:
            overall_status = ComplianceStatus.COMPLIANT
        elif overall_score >= 0.7:
            overall_status = ComplianceStatus.AT_RISK
        else:
            overall_status = ComplianceStatus.NON_COMPLIANT

        # Count requirements by status
        status_counts = {
            "total": len(requirement_results),
            "compliant": sum(
                1
                for r in requirement_results.values()
                if r["status"] == ComplianceStatus.COMPLIANT
            ),
            "non_compliant": sum(
                1
                for r in requirement_results.values()
                if r["status"] == ComplianceStatus.NON_COMPLIANT
            ),
            "at_risk": sum(
                1
                for r in requirement_results.values()
                if r["status"] == ComplianceStatus.AT_RISK
            ),
            "unknown": sum(
                1
                for r in requirement_results.values()
                if r["status"] == ComplianceStatus.UNKNOWN
            ),
        }

        # Generate recommendations
        recommendations = self._generate_recommendations(requirement_results)

        return ComplianceReport(
            report_id=report_id,
            generated_at=datetime.utcnow(),
            period_start=start_date,
            period_end=end_date,
            overall_status=overall_status,
            compliance_score=overall_score,
            total_requirements=status_counts["total"],
            met_requirements=status_counts["compliant"],
            failed_requirements=status_counts["non_compliant"],
            at_risk_requirements=status_counts["at_risk"],
            details=requirement_results,
            recommendations=recommendations,
        )

    def _check_audit_logging(
        self, start_date: datetime, end_date: datetime
    ) -> Dict[str, Any]:
        """Check audit logging compliance.

        What audit logging compliance provides:
        - Immutable audit trail verification
        - Complete change tracking
        - Tamper-proof evidence
        - Regulatory compliance
        """
        try:
            # Check if immutable audit logs exist
            result = self.db.execute(
                text(
                    """
                SELECT COUNT(*) as total_records,
                       COUNT(CASE WHEN is_immutable = true THEN 1 END) as immutable_records
                FROM immutable_audit_logs
                WHERE created_at >= :start_date AND created_at <= :end_date
            """
                ),
                {"start_date": start_date, "end_date": end_date},
            ).fetchone()

            total_records = result.total_records
            immutable_records = result.immutable_records

            # Check audit chain integrity
            integrity_result = self.db.execute(
                text(
                    """
                SELECT COUNT(*) as total_checks,
                       COUNT(CASE WHEN is_valid = true THEN 1 END) as valid_checks
                FROM verify_audit_chain_integrity()
            """
                )
            ).fetchone()

            total_checks = integrity_result.total_checks
            valid_checks = integrity_result.valid_checks

            # Calculate compliance score
            if total_records == 0:
                score = 0.0
                status = ComplianceStatus.NON_COMPLIANT
            elif immutable_records == total_records and valid_checks == total_checks:
                score = 1.0
                status = ComplianceStatus.COMPLIANT
            elif (
                immutable_records >= total_records * 0.95
                and valid_checks >= total_checks * 0.95
            ):
                score = 0.8
                status = ComplianceStatus.AT_RISK
            else:
                score = 0.3
                status = ComplianceStatus.NON_COMPLIANT

            return {
                "status": status,
                "score": score,
                "details": {
                    "total_audit_records": total_records,
                    "immutable_records": immutable_records,
                    "audit_chain_checks": total_checks,
                    "valid_chain_checks": valid_checks,
                    "immutability_percentage": (
                        (immutable_records / total_records * 100)
                        if total_records > 0
                        else 0
                    ),
                    "chain_integrity_percentage": (
                        (valid_checks / total_checks * 100) if total_checks > 0 else 0
                    ),
                },
                "recommendations": self._get_audit_logging_recommendations(score),
            }

        except Exception as e:
            logger.error("Audit logging check failed: %s", e)
            return {
                "status": ComplianceStatus.UNKNOWN,
                "score": 0.0,
                "details": {"error": str(e)},
                "recommendations": ["Fix audit logging system"],
            }

    def _check_data_retention(
        self, start_date: datetime, end_date: datetime
    ) -> Dict[str, Any]:
        """Check data retention compliance.

        What data retention compliance provides:
        - Policy adherence verification
        - Automated cleanup validation
        - Storage optimization
        - Regulatory compliance
        """
        try:
            # Check data retention policies
            retention_checks = {
                "audit_logs_retention": self._check_table_retention(
                    "audit_logs", 730
                ),  # 2 years
                "access_logs_retention": self._check_table_retention(
                    "access_logs", 365
                ),  # 1 year
                "user_activities_retention": self._check_table_retention(
                    "user_activities", 365
                ),  # 1 year
                "sessions_retention": self._check_table_retention(
                    "sessions", 7
                ),  # 7 days
            }

            # Calculate overall score
            total_checks = len(retention_checks)
            passed_checks = sum(
                1 for check in retention_checks.values() if check["compliant"]
            )
            score = passed_checks / total_checks

            if score >= 0.9:
                status = ComplianceStatus.COMPLIANT
            elif score >= 0.7:
                status = ComplianceStatus.AT_RISK
            else:
                status = ComplianceStatus.NON_COMPLIANT

            return {
                "status": status,
                "score": score,
                "details": retention_checks,
                "recommendations": self._get_retention_recommendations(score),
            }

        except Exception as e:
            logger.error("Data retention check failed: %s", e)
            return {
                "status": ComplianceStatus.UNKNOWN,
                "score": 0.0,
                "details": {"error": str(e)},
                "recommendations": ["Fix data retention system"],
            }

    def _check_user_authentication(
        self, start_date: datetime, end_date: datetime
    ) -> Dict[str, Any]:
        """Check user authentication compliance.

        What user authentication compliance provides:
        - Security policy enforcement
        - Access control validation
        - Authentication mechanism verification
        - Authorization compliance
        """
        try:
            # Check user authentication metrics
            auth_stats = self.db.execute(
                text(
                    """
                SELECT 
                    COUNT(*) as total_users,
                    COUNT(CASE WHEN is_active = true THEN 1 END) as active_users,
                    COUNT(CASE WHEN password_hash IS NOT NULL THEN 1 END) as users_with_passwords,
                    COUNT(CASE WHEN role = 'admin' THEN 1 END) as admin_users
                FROM users
            """
                )
            ).fetchone()

            # Check recent authentication activities
            recent_auth = self.db.execute(
                text(
                    """
                SELECT COUNT(*) as auth_activities
                FROM user_activities
                WHERE activity_type IN ('user_login', 'user_logout', 'password_change')
                AND created_at >= :start_date AND created_at <= :end_date
            """
                ),
                {"start_date": start_date, "end_date": end_date},
            ).fetchone()

            # Calculate compliance score
            score = 0.0
            if auth_stats.total_users > 0:
                score += 0.3 if auth_stats.active_users > 0 else 0
                score += (
                    0.3
                    if auth_stats.users_with_passwords == auth_stats.total_users
                    else 0
                )
                score += 0.2 if auth_stats.admin_users > 0 else 0
                score += 0.2 if recent_auth.auth_activities > 0 else 0

            if score >= 0.9:
                status = ComplianceStatus.COMPLIANT
            elif score >= 0.7:
                status = ComplianceStatus.AT_RISK
            else:
                status = ComplianceStatus.NON_COMPLIANT

            return {
                "status": status,
                "score": score,
                "details": {
                    "total_users": auth_stats.total_users,
                    "active_users": auth_stats.active_users,
                    "users_with_passwords": auth_stats.users_with_passwords,
                    "admin_users": auth_stats.admin_users,
                    "recent_auth_activities": recent_auth.auth_activities,
                },
                "recommendations": self._get_auth_recommendations(score),
            }

        except Exception as e:
            logger.error("User authentication check failed: %s", e)
            return {
                "status": ComplianceStatus.UNKNOWN,
                "score": 0.0,
                "details": {"error": str(e)},
                "recommendations": ["Fix authentication system"],
            }

    def _check_data_encryption(
        self, start_date: datetime, end_date: datetime
    ) -> Dict[str, Any]:
        """Check data encryption compliance.

        What data encryption compliance provides:
        - Encryption at rest verification
        - Encryption in transit validation
        - Key management compliance
        - Data protection standards
        """
        try:
            # Check if encryption is properly configured
            encryption_checks = {
                "database_encryption": self._check_database_encryption(),
                "application_encryption": self._check_application_encryption(),
                "key_management": self._check_key_management(),
            }

            # Calculate overall score
            total_checks = len(encryption_checks)
            passed_checks = sum(
                1 for check in encryption_checks.values() if check["compliant"]
            )
            score = passed_checks / total_checks

            if score >= 0.9:
                status = ComplianceStatus.COMPLIANT
            elif score >= 0.7:
                status = ComplianceStatus.AT_RISK
            else:
                status = ComplianceStatus.NON_COMPLIANT

            return {
                "status": status,
                "score": score,
                "details": encryption_checks,
                "recommendations": self._get_encryption_recommendations(score),
            }

        except Exception as e:
            logger.error("Data encryption check failed: %s", e)
            return {
                "status": ComplianceStatus.UNKNOWN,
                "score": 0.0,
                "details": {"error": str(e)},
                "recommendations": ["Fix encryption system"],
            }

    def _check_access_control(
        self, start_date: datetime, end_date: datetime
    ) -> Dict[str, Any]:
        """Check access control compliance.

        What access control compliance provides:
        - Role-based access validation
        - Permission enforcement verification
        - Authorization compliance
        - Security policy adherence
        """
        try:
            # Check role-based access control
            rbac_stats = self.db.execute(
                text(
                    """
                SELECT 
                    COUNT(DISTINCT role) as total_roles,
                    COUNT(*) as total_permissions,
                    COUNT(DISTINCT user_id) as users_with_roles
                FROM permissions p
                LEFT JOIN users u ON u.role = p.role
            """
                )
            ).fetchone()

            # Check recent access control activities
            recent_access = self.db.execute(
                text(
                    """
                SELECT COUNT(*) as access_activities
                FROM access_logs
                WHERE created_at >= :start_date AND created_at <= :end_date
            """
                ),
                {"start_date": start_date, "end_date": end_date},
            ).fetchone()

            # Calculate compliance score
            score = 0.0
            if rbac_stats.total_roles > 0:
                score += 0.4 if rbac_stats.total_permissions > 0 else 0
                score += 0.3 if rbac_stats.users_with_roles > 0 else 0
                score += 0.3 if recent_access.access_activities > 0 else 0

            if score >= 0.9:
                status = ComplianceStatus.COMPLIANT
            elif score >= 0.7:
                status = ComplianceStatus.AT_RISK
            else:
                status = ComplianceStatus.NON_COMPLIANT

            return {
                "status": status,
                "score": score,
                "details": {
                    "total_roles": rbac_stats.total_roles,
                    "total_permissions": rbac_stats.total_permissions,
                    "users_with_roles": rbac_stats.users_with_roles,
                    "recent_access_activities": recent_access.access_activities,
                },
                "recommendations": self._get_access_control_recommendations(score),
            }

        except Exception as e:
            logger.error("Access control check failed: %s", e)
            return {
                "status": ComplianceStatus.UNKNOWN,
                "score": 0.0,
                "details": {"error": str(e)},
                "recommendations": ["Fix access control system"],
            }

    def _check_table_retention(
        self, table_name: str, retention_days: int
    ) -> Dict[str, Any]:
        """Check if table data retention is compliant."""
        try:
            # Check if old data exists beyond retention period
            old_data_count = self.db.execute(
                text(
                    f"""
                SELECT COUNT(*) as old_records
                FROM {table_name}
                WHERE created_at < NOW() - INTERVAL '{retention_days} days'
            """
                )
            ).fetchone()

            compliant = old_data_count.old_records == 0
            return {
                "compliant": compliant,
                "old_records": old_data_count.old_records,
                "retention_days": retention_days,
            }
        except Exception as e:
            logger.error("Table retention check failed for %s: %s", table_name, e)
            return {"compliant": False, "error": str(e)}

    def _check_database_encryption(self) -> Dict[str, Any]:
        """Check database encryption status."""
        try:
            # Check if database is configured for encryption
            result = self.db.execute(
                text(
                    """
                SELECT 
                    CASE WHEN setting LIKE '%ssl%' THEN true ELSE false END as ssl_enabled,
                    setting as ssl_setting
                FROM pg_settings 
                WHERE name = 'ssl'
            """
                )
            ).fetchone()

            return {
                "compliant": result.ssl_enabled if result else False,
                "ssl_enabled": result.ssl_enabled if result else False,
                "ssl_setting": result.ssl_setting if result else "Not configured",
            }
        except Exception as e:
            return {"compliant": False, "error": str(e)}

    def _check_application_encryption(self) -> Dict[str, Any]:
        """Check application-level encryption."""
        import os

        return {
            "compliant": bool(os.environ.get("ENCRYPTION_KEY")),
            "encryption_key_configured": bool(os.environ.get("ENCRYPTION_KEY")),
            "secret_key_configured": bool(os.environ.get("SECRET_KEY")),
        }

    def _check_key_management(self) -> Dict[str, Any]:
        """Check key management compliance."""
        import os

        return {
            "compliant": bool(os.environ.get("SECRET_KEY"))
            and bool(os.environ.get("ENCRYPTION_KEY")),
            "secret_key_configured": bool(os.environ.get("SECRET_KEY")),
            "encryption_key_configured": bool(os.environ.get("ENCRYPTION_KEY")),
        }

    def _generate_recommendations(
        self, requirement_results: Dict[str, Any]
    ) -> List[str]:
        """Generate compliance recommendations."""
        recommendations = []

        for req_id, result in requirement_results.items():
            if result["status"] == ComplianceStatus.NON_COMPLIANT:
                recommendations.extend(result.get("recommendations", []))
            elif result["status"] == ComplianceStatus.AT_RISK:
                recommendations.extend([f"Monitor {req_id} closely"])

        return list(set(recommendations))  # Remove duplicates

    def _get_audit_logging_recommendations(self, score: float) -> List[str]:
        """Get audit logging recommendations."""
        if score < 0.5:
            return ["Implement immutable audit logging", "Fix audit chain integrity"]
        elif score < 0.8:
            return ["Improve audit logging coverage", "Monitor audit chain integrity"]
        else:
            return ["Maintain current audit logging standards"]

    def _get_retention_recommendations(self, score: float) -> List[str]:
        """Get data retention recommendations."""
        if score < 0.5:
            return ["Implement data retention policies", "Set up automated cleanup"]
        elif score < 0.8:
            return ["Review data retention policies", "Optimize cleanup schedules"]
        else:
            return ["Maintain current retention policies"]

    def _get_auth_recommendations(self, score: float) -> List[str]:
        """Get authentication recommendations."""
        if score < 0.5:
            return ["Implement proper user authentication", "Set up password policies"]
        elif score < 0.8:
            return ["Review authentication policies", "Monitor user activities"]
        else:
            return ["Maintain current authentication standards"]

    def _get_encryption_recommendations(self, score: float) -> List[str]:
        """Get encryption recommendations."""
        if score < 0.5:
            return ["Implement data encryption", "Configure encryption keys"]
        elif score < 0.8:
            return ["Review encryption configuration", "Update encryption keys"]
        else:
            return ["Maintain current encryption standards"]

    def _get_access_control_recommendations(self, score: float) -> List[str]:
        """Get access control recommendations."""
        if score < 0.5:
            return ["Implement role-based access control", "Set up permissions"]
        elif score < 0.8:
            return ["Review access control policies", "Monitor user permissions"]
        else:
            return ["Maintain current access control standards"]


def generate_compliance_report(
    start_date: Optional[datetime] = None, end_date: Optional[datetime] = None
) -> ComplianceReport:
    """Generate a compliance report."""
    db = next(get_db())
    manager = ComplianceManager(db)
    return manager.check_compliance_status(start_date, end_date)


def check_compliance_status() -> ComplianceStatus:
    """Check current compliance status."""
    report = generate_compliance_report()
    return report.overall_status


def get_compliance_metrics() -> Dict[str, Any]:
    """Get compliance metrics."""
    report = generate_compliance_report()
    return {
        "overall_status": report.overall_status.value,
        "compliance_score": report.compliance_score,
        "total_requirements": report.total_requirements,
        "met_requirements": report.met_requirements,
        "failed_requirements": report.failed_requirements,
        "at_risk_requirements": report.at_risk_requirements,
    }
