"""Database models for MPS Connect using SQLAlchemy."""

from uuid import uuid4

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Integer,
    String,
    Text,
    ForeignKey,
    Index,
    CheckConstraint,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID as PostgresUUID, JSONB, INET
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

Base = declarative_base()


class User(Base):
    """User model for authentication and authorization."""

    __tablename__ = "users"

    id = Column(PostgresUUID(as_uuid=True), primary_key=True, default=uuid4)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    name = Column(String(255), nullable=False)
    role = Column(String(50), default="user", nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    is_active = Column(Boolean, default=True, nullable=False)

    # Relationships
    cases = relationship("Case", back_populates="user", cascade="all, delete-orphan")
    sessions = relationship(
        "Session", back_populates="user", cascade="all, delete-orphan"
    )
    activities = relationship(
        "UserActivity", back_populates="user", cascade="all, delete-orphan"
    )

    __table_args__ = (
        CheckConstraint(
            "role IN ('user', 'admin', 'mp_staff')", name="check_user_role"
        ),
    )


class Case(Base):
    """Case model for constituent cases."""

    __tablename__ = "cases"

    id = Column(PostgresUUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(
        PostgresUUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    case_type = Column(String(100), nullable=False)
    status = Column(String(50), default="draft", nullable=False)
    priority = Column(String(20), default="medium", nullable=False)
    subject = Column(Text, nullable=False)
    description = Column(Text)
    constituent_info = Column(JSONB)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    submitted_at = Column(DateTime(timezone=True))
    reviewed_at = Column(DateTime(timezone=True))

    # Relationships
    user = relationship("User", back_populates="cases")
    conversations = relationship(
        "Conversation", back_populates="case", cascade="all, delete-orphan"
    )
    letters = relationship(
        "Letter", back_populates="case", cascade="all, delete-orphan"
    )

    __table_args__ = (
        CheckConstraint(
            "status IN ('draft', 'submitted', 'under_review', 'approved', 'rejected')",
            name="check_case_status",
        ),
        CheckConstraint(
            "priority IN ('low', 'medium', 'high', 'urgent')",
            name="check_case_priority",
        ),
        Index("idx_cases_user_id", "user_id"),
        Index("idx_cases_status", "status"),
        Index("idx_cases_created_at", "created_at"),
    )


class Conversation(Base):
    """Conversation model for chat interactions."""

    __tablename__ = "conversations"

    id = Column(PostgresUUID(as_uuid=True), primary_key=True, default=uuid4)
    case_id = Column(
        PostgresUUID(as_uuid=True),
        ForeignKey("cases.id", ondelete="CASCADE"),
        nullable=False,
    )
    session_id = Column(String(255), nullable=False)
    step_number = Column(Integer, nullable=False)
    question = Column(Text, nullable=False)
    answer = Column(Text)
    extracted_facts = Column(JSONB)
    confidence_score = Column(String(10))  # DECIMAL(5,4) as string for SQLAlchemy
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    case = relationship("Case", back_populates="conversations")

    __table_args__ = (
        Index("idx_conversations_case_id", "case_id"),
        Index("idx_conversations_session_id", "session_id"),
    )


class Letter(Base):
    """Letter model for generated letters."""

    __tablename__ = "letters"

    id = Column(PostgresUUID(as_uuid=True), primary_key=True, default=uuid4)
    case_id = Column(
        PostgresUUID(as_uuid=True),
        ForeignKey("cases.id", ondelete="CASCADE"),
        nullable=False,
    )
    letter_type = Column(String(50), nullable=False)
    recipient_agency = Column(String(255), nullable=False)
    subject_line = Column(Text, nullable=False)
    content = Column(Text, nullable=False)
    status = Column(String(50), default="draft", nullable=False)
    generated_at = Column(DateTime(timezone=True), server_default=func.now())
    reviewed_at = Column(DateTime(timezone=True))
    approved_at = Column(DateTime(timezone=True))
    sent_at = Column(DateTime(timezone=True))
    letter_metadata = Column(JSONB)

    # Relationships
    case = relationship("Case", back_populates="letters")

    __table_args__ = (
        CheckConstraint(
            "letter_type IN ('standard', 'urgent', 'compassionate', 'formal', 'follow_up')",
            name="check_letter_type",
        ),
        CheckConstraint(
            "status IN ('draft', 'reviewed', 'approved', 'sent')",
            name="check_letter_status",
        ),
        Index("idx_letters_case_id", "case_id"),
        Index("idx_letters_status", "status"),
    )


class AuditLog(Base):
    """Audit log model for tracking changes."""

    __tablename__ = "audit_logs"

    id = Column(PostgresUUID(as_uuid=True), primary_key=True, default=uuid4)
    table_name = Column(String(100), nullable=False)
    record_id = Column(PostgresUUID(as_uuid=True), nullable=False)
    action = Column(String(20), nullable=False)
    old_values = Column(JSONB)
    new_values = Column(JSONB)
    user_id = Column(PostgresUUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    ip_address = Column(INET)
    user_agent = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        CheckConstraint(
            "action IN ('INSERT', 'UPDATE', 'DELETE')", name="check_audit_action"
        ),
        Index("idx_audit_logs_table_record", "table_name", "record_id"),
        Index("idx_audit_logs_created_at", "created_at"),
    )


class DataLineage(Base):
    """Data lineage model for tracking data relationships."""

    __tablename__ = "data_lineage"

    id = Column(PostgresUUID(as_uuid=True), primary_key=True, default=uuid4)
    source_table = Column(String(100), nullable=False)
    source_id = Column(PostgresUUID(as_uuid=True), nullable=False)
    target_table = Column(String(100), nullable=False)
    target_id = Column(PostgresUUID(as_uuid=True), nullable=False)
    relationship_type = Column(String(50), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class UserActivity(Base):
    """User activity model for tracking user actions."""

    __tablename__ = "user_activities"

    id = Column(PostgresUUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(
        PostgresUUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    activity_type = Column(String(100), nullable=False)
    description = Column(Text)
    ip_address = Column(INET)
    user_agent = Column(Text)
    activity_metadata = Column(JSONB)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    user = relationship("User", back_populates="activities")

    __table_args__ = (Index("idx_user_activities_user_id", "user_id"),)


class Session(Base):
    """Session model for user authentication."""

    __tablename__ = "sessions"

    id = Column(PostgresUUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(
        PostgresUUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    session_token = Column(String(255), unique=True, nullable=False, index=True)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_accessed_at = Column(DateTime(timezone=True), server_default=func.now())
    is_active = Column(Boolean, default=True, nullable=False)

    # Relationships
    user = relationship("User", back_populates="sessions")

    __table_args__ = (Index("idx_sessions_user_id", "user_id"),)


class Permission(Base):
    """Permission model for role-based access control."""

    __tablename__ = "permissions"

    id = Column(PostgresUUID(as_uuid=True), primary_key=True, default=uuid4)
    role = Column(String(50), nullable=False)
    resource = Column(String(100), nullable=False)
    action = Column(String(50), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        UniqueConstraint(
            "role", "resource", "action", name="unique_role_resource_action"
        ),
    )


class AccessLog(Base):
    """Access log model for API request tracking."""

    __tablename__ = "access_logs"

    id = Column(PostgresUUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(
        PostgresUUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )
    endpoint = Column(String(255), nullable=False)
    method = Column(String(10), nullable=False)
    status_code = Column(Integer, nullable=False)
    response_time_ms = Column(Integer)
    ip_address = Column(INET)
    user_agent = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("idx_access_logs_user_id", "user_id"),
        Index("idx_access_logs_created_at", "created_at"),
    )
