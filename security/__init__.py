"""Security module for MPS Connect."""

from .auth import (
    create_access_token,
    verify_token,
    get_current_user,
    authenticate_user,
    create_user,
    hash_password,
    verify_password,
    require_auth,
    require_role,
)
from .encryption import (
    encrypt_data,
    decrypt_data,
    generate_key,
    hash_sensitive_data,
)
from .audit import (
    log_user_activity,
    log_api_access,
    log_data_change,
    get_audit_trail,
)

__all__ = [
    "create_access_token",
    "verify_token",
    "get_current_user",
    "authenticate_user",
    "create_user",
    "hash_password",
    "verify_password",
    "require_auth",
    "require_role",
    "encrypt_data",
    "decrypt_data",
    "generate_key",
    "hash_sensitive_data",
    "log_user_activity",
    "log_api_access",
    "log_data_change",
    "get_audit_trail",
]
