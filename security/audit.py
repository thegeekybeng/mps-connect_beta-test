"""Audit logging and compliance tracking for MPS Connect."""

import logging
import json
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy.orm import Session
from fastapi import Request

from database.connection import get_db
from database.models import AuditLog, UserActivity, AccessLog, User

logger = logging.getLogger(__name__)


def log_user_activity(
    db: Session,
    user_id: UUID,
    activity_type: str,
    description: str,
    metadata: Optional[Dict[str, Any]] = None,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None,
) -> UserActivity:
    """Log user activity for audit trail.

    Args:
        db: Database session
        user_id: User ID
        activity_type: Type of activity
        description: Activity description
        metadata: Additional metadata
        ip_address: User IP address
        user_agent: User agent string

    Returns:
        Created UserActivity record

    What audit logging provides:
    - Complete user action tracking
    - Compliance with government regulations
    - Security incident investigation
    - User behavior analysis
    """
    activity = UserActivity(
        user_id=user_id,
        activity_type=activity_type,
        description=description,
        metadata=metadata or {},
        ip_address=ip_address,
        user_agent=user_agent,
    )

    db.add(activity)
    db.commit()
    db.refresh(activity)

    logger.info("User activity logged: %s - %s", activity_type, description)
    return activity


def log_api_access(
    db: Session,
    user_id: Optional[UUID],
    endpoint: str,
    method: str,
    status_code: int,
    response_time_ms: int,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None,
) -> AccessLog:
    """Log API access for monitoring and security.

    Args:
        db: Database session
        user_id: User ID (None for anonymous access)
        endpoint: API endpoint
        method: HTTP method
        status_code: Response status code
        response_time_ms: Response time in milliseconds
        ip_address: Client IP address
        user_agent: Client user agent

    Returns:
        Created AccessLog record

    What API logging provides:
    - Performance monitoring
    - Security threat detection
    - Usage analytics
    - Error tracking
    """
    access_log = AccessLog(
        user_id=user_id,
        endpoint=endpoint,
        method=method,
        status_code=status_code,
        response_time_ms=response_time_ms,
        ip_address=ip_address,
        user_agent=user_agent,
    )

    db.add(access_log)
    db.commit()
    db.refresh(access_log)

    logger.info("API access logged: %s %s - %d", method, endpoint, status_code)
    return access_log


def log_data_change(
    db: Session,
    table_name: str,
    record_id: UUID,
    action: str,
    old_values: Optional[Dict[str, Any]] = None,
    new_values: Optional[Dict[str, Any]] = None,
    user_id: Optional[UUID] = None,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None,
) -> AuditLog:
    """Log data changes for audit trail.

    Args:
        db: Database session
        table_name: Name of changed table
        record_id: ID of changed record
        action: Type of change (INSERT, UPDATE, DELETE)
        old_values: Previous values (for UPDATE/DELETE)
        new_values: New values (for INSERT/UPDATE)
        user_id: User who made the change
        ip_address: User IP address
        user_agent: User agent string

    Returns:
        Created AuditLog record

    What data change logging provides:
    - Immutable audit trail
    - Data integrity verification
    - Compliance requirements
    - Change tracking and rollback capability
    """
    audit_log = AuditLog(
        table_name=table_name,
        record_id=record_id,
        action=action,
        old_values=old_values or {},
        new_values=new_values or {},
        user_id=user_id,
        ip_address=ip_address,
        user_agent=user_agent,
    )

    db.add(audit_log)
    db.commit()
    db.refresh(audit_log)

    logger.info("Data change logged: %s %s on %s", action, record_id, table_name)
    return audit_log


def get_audit_trail(
    db: Session,
    table_name: Optional[str] = None,
    record_id: Optional[UUID] = None,
    user_id: Optional[UUID] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    limit: int = 100,
) -> List[AuditLog]:
    """Get audit trail for compliance reporting.

    Args:
        db: Database session
        table_name: Filter by table name
        record_id: Filter by record ID
        user_id: Filter by user ID
        start_date: Filter by start date
        end_date: Filter by end date
        limit: Maximum number of records

    Returns:
        List of AuditLog records

    What audit trail provides:
    - Complete change history
    - Compliance reporting
    - Security investigation
    - Data lineage tracking
    """
    query = db.query(AuditLog)

    if table_name:
        query = query.filter(AuditLog.table_name == table_name)

    if record_id:
        query = query.filter(AuditLog.record_id == record_id)

    if user_id:
        query = query.filter(AuditLog.user_id == user_id)

    if start_date:
        query = query.filter(AuditLog.created_at >= start_date)

    if end_date:
        query = query.filter(AuditLog.created_at <= end_date)

    return query.order_by(AuditLog.created_at.desc()).limit(limit).all()


def get_user_activities(
    db: Session,
    user_id: Optional[UUID] = None,
    activity_type: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    limit: int = 100,
) -> List[UserActivity]:
    """Get user activities for analysis.

    Args:
        db: Database session
        user_id: Filter by user ID
        activity_type: Filter by activity type
        start_date: Filter by start date
        end_date: Filter by end date
        limit: Maximum number of records

    Returns:
        List of UserActivity records
    """
    query = db.query(UserActivity)

    if user_id:
        query = query.filter(UserActivity.user_id == user_id)

    if activity_type:
        query = query.filter(UserActivity.activity_type == activity_type)

    if start_date:
        query = query.filter(UserActivity.created_at >= start_date)

    if end_date:
        query = query.filter(UserActivity.created_at <= end_date)

    return query.order_by(UserActivity.created_at.desc()).limit(limit).all()


def get_api_access_stats(
    db: Session,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Get API access statistics for monitoring.

    Args:
        db: Database session
        start_date: Filter by start date
        end_date: Filter by end date

    Returns:
        Dictionary with access statistics

    What API stats provide:
    - Performance metrics
    - Usage patterns
    - Error rates
    - Security insights
    """
    from sqlalchemy import func

    query = db.query(AccessLog)

    if start_date:
        query = query.filter(AccessLog.created_at >= start_date)

    if end_date:
        query = query.filter(AccessLog.created_at <= end_date)

    # Get basic stats
    total_requests = query.count()

    # Status code distribution
    status_stats = (
        query.with_entities(
            AccessLog.status_code, func.count(AccessLog.id).label("count")
        )
        .group_by(AccessLog.status_code)
        .all()
    )

    # Average response time
    avg_response_time = (
        query.with_entities(func.avg(AccessLog.response_time_ms)).scalar() or 0
    )

    # Top endpoints
    top_endpoints = (
        query.with_entities(AccessLog.endpoint, func.count(AccessLog.id).label("count"))
        .group_by(AccessLog.endpoint)
        .order_by(func.count(AccessLog.id).desc())
        .limit(10)
        .all()
    )

    return {
        "total_requests": total_requests,
        "status_code_distribution": {str(s.status_code): s.count for s in status_stats},
        "average_response_time_ms": round(avg_response_time, 2),
        "top_endpoints": [
            {"endpoint": e.endpoint, "count": e.count} for e in top_endpoints
        ],
    }


def generate_compliance_report(
    db: Session, start_date: datetime, end_date: datetime
) -> Dict[str, Any]:
    """Generate compliance report for audit purposes.

    Args:
        db: Database session
        start_date: Report start date
        end_date: Report end date

    Returns:
        Compliance report data

    What compliance reporting provides:
    - Regulatory compliance documentation
    - Audit trail verification
    - Security posture assessment
    - Data governance metrics
    """
    # Get audit logs
    audit_logs = get_audit_trail(
        db, start_date=start_date, end_date=end_date, limit=1000
    )

    # Get user activities
    user_activities = get_user_activities(
        db, start_date=start_date, end_date=end_date, limit=1000
    )

    # Get API stats
    api_stats = get_api_access_stats(db, start_date=start_date, end_date=end_date)

    # Count by action type
    action_counts = {}
    for log in audit_logs:
        action_counts[log.action] = action_counts.get(log.action, 0) + 1

    # Count by activity type
    activity_counts = {}
    for activity in user_activities:
        activity_counts[activity.activity_type] = (
            activity_counts.get(activity.activity_type, 0) + 1
        )

    return {
        "report_period": {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
        },
        "audit_summary": {
            "total_changes": len(audit_logs),
            "action_breakdown": action_counts,
            "tables_affected": list(set(log.table_name for log in audit_logs)),
        },
        "user_activity_summary": {
            "total_activities": len(user_activities),
            "activity_breakdown": activity_counts,
            "unique_users": len(set(activity.user_id for activity in user_activities)),
        },
        "api_usage": api_stats,
        "compliance_status": {
            "audit_logging_active": len(audit_logs) > 0,
            "user_tracking_active": len(user_activities) > 0,
            "api_monitoring_active": api_stats["total_requests"] > 0,
        },
    }


def cleanup_old_audit_data(db: Session, days_to_keep: int = 730) -> Dict[str, int]:
    """Clean up old audit data according to retention policy.

    Args:
        db: Database session
        days_to_keep: Number of days to keep data

    Returns:
        Dictionary with cleanup statistics

    What cleanup provides:
    - Storage optimization
    - Compliance with retention policies
    - Performance improvement
    - Cost reduction
    """
    from datetime import timedelta

    cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)

    # Clean up old audit logs
    old_audit_logs = db.query(AuditLog).filter(AuditLog.created_at < cutoff_date).all()
    audit_count = len(old_audit_logs)

    for log in old_audit_logs:
        db.delete(log)

    # Clean up old user activities
    old_activities = (
        db.query(UserActivity).filter(UserActivity.created_at < cutoff_date).all()
    )
    activity_count = len(old_activities)

    for activity in old_activities:
        db.delete(activity)

    # Clean up old access logs
    old_access_logs = (
        db.query(AccessLog).filter(AccessLog.created_at < cutoff_date).all()
    )
    access_count = len(old_access_logs)

    for log in old_access_logs:
        db.delete(log)

    db.commit()

    logger.info(
        "Audit data cleanup completed: %d audit logs, %d activities, %d access logs removed",
        audit_count,
        activity_count,
        access_count,
    )

    return {
        "audit_logs_removed": audit_count,
        "user_activities_removed": activity_count,
        "access_logs_removed": access_count,
        "cutoff_date": cutoff_date.isoformat(),
    }
