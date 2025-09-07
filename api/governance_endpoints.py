"""Governance endpoints for MPS Connect compliance and audit management."""

# pylint: disable=import-error,no-name-in-module
import logging
from datetime import datetime
from typing import Any, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from ..database.connection import get_db  # type: ignore
from ..database.models import User  # type: ignore
from ..security.auth import require_role  # type: ignore
from ..governance.compliance import (  # type: ignore
    ComplianceStatus,
    generate_compliance_report,
    check_compliance_status,
    get_compliance_metrics as get_compliance_metrics_func,
)
from ..governance.audit_verification import (  # type: ignore
    AuditVerifier,
    check_audit_integrity,
    get_audit_summary as get_audit_summary_func,
    validate_audit_records as validate_audit_records_func,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/governance", tags=["governance"])


# Pydantic models
class ComplianceReportResponse(BaseModel):
    """Compliance report response model."""

    report_id: str
    generated_at: str
    period_start: str
    period_end: str
    overall_status: str
    compliance_score: float
    total_requirements: int
    met_requirements: int
    failed_requirements: int
    at_risk_requirements: int
    details: dict
    recommendations: List[str]


class AuditIntegrityResponse(BaseModel):
    """Audit integrity response model."""

    status: str
    risk_level: str
    total_records: int
    integrity_percentage: float
    broken_links_count: int
    verification_timestamp: str


@router.get("/compliance/status", response_model=ComplianceReportResponse)
async def get_compliance_status(
    start_date: Optional[datetime] = Query(
        None, description="Start date for compliance check"
    ),
    end_date: Optional[datetime] = Query(
        None, description="End date for compliance check"
    ),
    current_user: User = Depends(require_role("admin")),
):
    """Get compliance status and generate report.

    What compliance status provides:
    - Overall compliance assessment
    - Requirement-by-requirement analysis
    - Risk identification and scoring
    - Regulatory compliance verification
    - Actionable recommendations
    """
    try:
        # Generate compliance report
        report = generate_compliance_report(start_date, end_date)

        return ComplianceReportResponse(
            report_id=report.report_id,
            generated_at=report.generated_at.isoformat(),
            period_start=report.period_start.isoformat(),
            period_end=report.period_end.isoformat(),
            overall_status=report.overall_status.value,
            compliance_score=report.compliance_score,
            total_requirements=report.total_requirements,
            met_requirements=report.met_requirements,
            failed_requirements=report.failed_requirements,
            at_risk_requirements=report.at_risk_requirements,
            details=report.details,
            recommendations=report.recommendations,
        )

    except Exception as e:
        logger.error("Compliance status check failed: %s", e)
        raise HTTPException(
            status_code=500, detail="Failed to generate compliance report"
        ) from e


@router.get("/compliance/metrics")
async def get_compliance_metrics(current_user: User = Depends(require_role("admin"))):
    """Get compliance metrics summary.

    What compliance metrics provide:
    - Quick compliance overview
    - Key performance indicators
    - Status summary
    - Risk indicators
    """
    try:
        metrics = get_compliance_metrics_func()
        return metrics

    except Exception as e:
        logger.error("Compliance metrics retrieval failed: %s", e)
        raise HTTPException(
            status_code=500, detail="Failed to retrieve compliance metrics"
        ) from e


@router.get("/audit/integrity", response_model=AuditIntegrityResponse)
async def get_audit_integrity(current_user: User = Depends(require_role("admin"))):
    """Check audit chain integrity.

    What audit integrity checking provides:
    - Chain of custody verification
    - Tamper detection
    - Data integrity validation
    - Forensic evidence verification
    - Compliance audit support
    """
    try:
        integrity_check = check_audit_integrity()

        return AuditIntegrityResponse(
            status=integrity_check["status"],
            risk_level=integrity_check["risk_level"],
            total_records=integrity_check["total_records"],
            integrity_percentage=integrity_check["integrity_percentage"],
            broken_links_count=integrity_check["broken_links_count"],
            verification_timestamp=integrity_check["verification_timestamp"],
        )

    except Exception as e:
        logger.error("Audit integrity check failed: %s", e)
        raise HTTPException(
            status_code=500, detail="Failed to check audit integrity"
        ) from e


@router.get("/audit/summary")
async def get_audit_summary(
    start_date: Optional[datetime] = Query(
        None, description="Start date for audit summary"
    ),
    end_date: Optional[datetime] = Query(
        None, description="End date for audit summary"
    ),
    current_user: User = Depends(require_role("admin")),
):
    """Get audit summary for specified period.

    What audit summary provides:
    - Period-based audit analysis
    - Activity patterns and trends
    - User activity tracking
    - Table activity breakdown
    - Compliance metrics
    """
    try:
        summary = get_audit_summary_func(start_date, end_date)
        return summary

    except Exception as e:
        logger.error("Audit summary generation failed: %s", e)
        raise HTTPException(
            status_code=500, detail="Failed to generate audit summary"
        ) from e


@router.post("/audit/validate")
async def validate_audit_records(
    record_ids: List[str], current_user: User = Depends(require_role("admin"))
):
    """Validate specific audit records.

    What audit record validation provides:
    - Individual record verification
    - Chain position validation
    - Integrity confirmation
    - Forensic evidence validation
    """
    try:
        if not record_ids:
            raise HTTPException(status_code=400, detail="No record IDs provided")

        validation_result = validate_audit_records_func(record_ids)
        return validation_result

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Audit record validation failed: %s", e)
        raise HTTPException(
            status_code=500, detail="Failed to validate audit records"
        ) from e


@router.get("/audit/chain-verification")
async def get_chain_verification_report(
    current_user: User = Depends(require_role("admin")),
):
    """Generate comprehensive chain verification report.

    What chain verification report provides:
    - Complete audit chain analysis
    - Integrity assessment
    - Risk evaluation
    - Compliance status
    - Detailed recommendations
    """
    try:
        db = next(get_db())
        verifier = AuditVerifier(db)
        report = verifier.get_chain_verification_report()
        return report

    except Exception as e:
        logger.error("Chain verification report generation failed: %s", e)
        raise HTTPException(
            status_code=500, detail="Failed to generate chain verification report"
        ) from e


@router.get("/audit/immutable-logs")
async def get_immutable_audit_logs(
    table_name: Optional[str] = Query(None, description="Filter by table name"),
    start_date: Optional[datetime] = Query(None, description="Start date filter"),
    end_date: Optional[datetime] = Query(None, description="End date filter"),
    limit: int = Query(100, description="Maximum number of records to return"),
    offset: int = Query(0, description="Number of records to skip"),
    current_user: User = Depends(require_role("admin")),
):
    """Get immutable audit logs with filtering.

    What immutable audit logs provide:
    - Tamper-proof audit trail
    - Complete change history
    - Chain of custody verification
    - Forensic evidence
    - Compliance documentation
    """
    try:
        from sqlalchemy import text

        # Build query with filters
        where_conditions = ["1=1"]
        params: dict[str, Any] = {"limit": limit, "offset": offset}

        if table_name:
            where_conditions.append("table_name = :table_name")
            params["table_name"] = table_name

        if start_date:
            where_conditions.append("created_at >= :start_date")
            params["start_date"] = start_date

        if end_date:
            where_conditions.append("created_at <= :end_date")
            params["end_date"] = end_date

        where_clause = " AND ".join(where_conditions)

        query = f"""
            SELECT 
                block_number,
                table_name,
                record_id,
                action,
                old_values,
                new_values,
                user_email,
                user_name,
                ip_address,
                user_agent,
                created_at,
                hash_chain,
                previous_hash
            FROM audit_logs_view
            WHERE {where_clause}
            ORDER BY block_number DESC
            LIMIT :limit OFFSET :offset
        """

        db = next(get_db())
        result = db.execute(text(query), params).fetchall()

        audit_logs = []
        for row in result:
            audit_logs.append(
                {
                    "block_number": row.block_number,
                    "table_name": row.table_name,
                    "record_id": str(row.record_id),
                    "action": row.action,
                    "old_values": row.old_values,
                    "new_values": row.new_values,
                    "user_email": row.user_email,
                    "user_name": row.user_name,
                    "ip_address": str(row.ip_address) if row.ip_address else None,
                    "user_agent": row.user_agent,
                    "created_at": row.created_at.isoformat(),
                    "hash_chain": row.hash_chain,
                    "previous_hash": row.previous_hash,
                }
            )

        return {
            "audit_logs": audit_logs,
            "total_returned": len(audit_logs),
            "limit": limit,
            "offset": offset,
            "filters": {
                "table_name": table_name,
                "start_date": start_date.isoformat() if start_date else None,
                "end_date": end_date.isoformat() if end_date else None,
            },
        }

    except Exception as e:
        logger.error("Immutable audit logs retrieval failed: %s", e)
        raise HTTPException(
            status_code=500, detail="Failed to retrieve immutable audit logs"
        ) from e


@router.get("/audit/chain-integrity")
async def get_chain_integrity_details(
    current_user: User = Depends(require_role("admin")),
):
    """Get detailed chain integrity information.

    What chain integrity details provide:
    - Complete chain verification results
    - Broken link identification
    - Integrity percentage calculation
    - Risk assessment
    - Forensic analysis support
    """
    try:
        db = next(get_db())
        verifier = AuditVerifier(db)
        integrity_result = verifier.verify_audit_chain()

        return {
            "is_valid": integrity_result.is_valid,
            "total_records": integrity_result.total_records,
            "valid_records": integrity_result.valid_records,
            "invalid_records": integrity_result.invalid_records,
            "chain_integrity_percentage": integrity_result.chain_integrity_percentage,
            "broken_links": integrity_result.broken_links,
            "first_record_timestamp": (
                integrity_result.first_record_timestamp.isoformat()
                if integrity_result.first_record_timestamp
                else None
            ),
            "last_record_timestamp": (
                integrity_result.last_record_timestamp.isoformat()
                if integrity_result.last_record_timestamp
                else None
            ),
            "verification_timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error("Chain integrity details retrieval failed: %s", e)
        raise HTTPException(
            status_code=500, detail="Failed to retrieve chain integrity details"
        ) from e


@router.get("/health")
async def get_governance_health(current_user: User = Depends(require_role("admin"))):
    """Get governance system health status.

    What governance health provides:
    - System status overview
    - Component health checks
    - Performance indicators
    - Alert conditions
    """
    try:
        # Check compliance status
        compliance_status = check_compliance_status()

        # Check audit integrity
        audit_integrity = check_audit_integrity()

        # Determine overall health
        if (
            compliance_status == ComplianceStatus.COMPLIANT
            and audit_integrity["status"] == "INTACT"
        ):
            health_status = "HEALTHY"
        elif (
            compliance_status == ComplianceStatus.AT_RISK
            or audit_integrity["status"] == "MINOR_CORRUPTION"
        ):
            health_status = "DEGRADED"
        else:
            health_status = "UNHEALTHY"

        return {
            "overall_status": health_status,
            "compliance_status": compliance_status.value,
            "audit_integrity_status": audit_integrity["status"],
            "audit_integrity_percentage": audit_integrity["integrity_percentage"],
            "total_audit_records": audit_integrity["total_records"],
            "broken_links_count": audit_integrity["broken_links_count"],
            "last_checked": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error("Governance health check failed: %s", e)
        raise HTTPException(
            status_code=500, detail="Failed to check governance health"
        ) from e
