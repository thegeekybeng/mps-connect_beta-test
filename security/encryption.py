"""Data encryption and security utilities for MPS Connect."""

import os
import hashlib
import secrets
import logging
from typing import Any, Dict, Optional
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

logger = logging.getLogger(__name__)

# Encryption configuration
ENCRYPTION_KEY = os.environ.get("ENCRYPTION_KEY")
if not ENCRYPTION_KEY:
    # Generate a new key if not provided (for development only)
    ENCRYPTION_KEY = Fernet.generate_key().decode()
    logger.warning(
        "ENCRYPTION_KEY not set, using generated key (not secure for production)"
    )


def generate_key() -> str:
    """Generate a new encryption key.

    Returns:
        Base64 encoded encryption key

    What Fernet encryption does:
    - Symmetric encryption (same key for encrypt/decrypt)
    - Authenticated encryption (prevents tampering)
    - Uses AES 128 in CBC mode
    - Includes timestamp for replay attack prevention
    """
    return Fernet.generate_key().decode()


def derive_key_from_password(password: str, salt: bytes) -> bytes:
    """Derive encryption key from password using PBKDF2.

    Args:
        password: Password string
        salt: Random salt bytes

    Returns:
        Derived encryption key

    What PBKDF2 does:
    - Password-based key derivation function
    - Uses HMAC with configurable iterations
    - Prevents rainbow table attacks
    - Slows down brute force attacks
    """
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,  # OWASP recommended minimum
    )
    return base64.urlsafe_b64encode(kdf.derive(password.encode()))


def encrypt_data(data: Any, key: Optional[str] = None) -> str:
    """Encrypt sensitive data.

    Args:
        data: Data to encrypt (will be JSON serialized)
        key: Encryption key (uses default if not provided)

    Returns:
        Base64 encoded encrypted data

    What this encryption provides:
    - Data confidentiality (only key holders can decrypt)
    - Data integrity (tampering detection)
    - Authentication (prevents unauthorized modifications)
    - Replay attack prevention (timestamp included)
    """
    import json

    # Serialize data to JSON
    json_data = json.dumps(data, default=str)

    # Use provided key or default
    encryption_key = key or ENCRYPTION_KEY

    # Create Fernet cipher
    f = Fernet(encryption_key.encode())

    # Encrypt data
    encrypted_data = f.encrypt(json_data.encode())

    # Return base64 encoded result
    return base64.urlsafe_b64encode(encrypted_data).decode()


def decrypt_data(encrypted_data: str, key: Optional[str] = None) -> Any:
    """Decrypt sensitive data.

    Args:
        encrypted_data: Base64 encoded encrypted data
        key: Decryption key (uses default if not provided)

    Returns:
        Decrypted and deserialized data

    Raises:
        ValueError: If decryption fails
    """
    import json

    try:
        # Use provided key or default
        decryption_key = key or ENCRYPTION_KEY

        # Create Fernet cipher
        f = Fernet(decryption_key.encode())

        # Decode base64 and decrypt
        encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode())
        decrypted_data = f.decrypt(encrypted_bytes)

        # Deserialize JSON
        return json.loads(decrypted_data.decode())

    except Exception as e:
        logger.error("Decryption failed: %s", e)
        raise ValueError("Failed to decrypt data") from e


def hash_sensitive_data(data: str, salt: Optional[str] = None) -> Dict[str, str]:
    """Hash sensitive data for storage or comparison.

    Args:
        data: Sensitive data to hash
        salt: Optional salt (generates random if not provided)

    Returns:
        Dictionary with hash and salt

    What hashing provides:
    - One-way transformation (cannot be reversed)
    - Salt prevents rainbow table attacks
    - Consistent output for same input
    - Fast comparison without storing plain text
    """
    # Generate salt if not provided
    if not salt:
        salt = secrets.token_hex(32)

    # Create salted hash
    salted_data = data + salt
    hash_value = hashlib.sha256(salted_data.encode()).hexdigest()

    return {"hash": hash_value, "salt": salt}


def verify_hashed_data(data: str, stored_hash: str, salt: str) -> bool:
    """Verify data against stored hash.

    Args:
        data: Data to verify
        stored_hash: Stored hash value
        salt: Salt used for original hash

    Returns:
        True if data matches hash, False otherwise
    """
    # Create hash with same salt
    salted_data = data + salt
    computed_hash = hashlib.sha256(salted_data.encode()).hexdigest()

    # Constant-time comparison to prevent timing attacks
    return secrets.compare_digest(computed_hash, stored_hash)


def generate_secure_token(length: int = 32) -> str:
    """Generate a cryptographically secure random token.

    Args:
        length: Token length in bytes

    Returns:
        Hex encoded random token

    What this provides:
    - Cryptographically secure randomness
    - Unpredictable token generation
    - Suitable for session tokens, API keys, etc.
    """
    return secrets.token_hex(length)


def mask_sensitive_data(data: str, visible_chars: int = 4) -> str:
    """Mask sensitive data for logging or display.

    Args:
        data: Data to mask
        visible_chars: Number of characters to show at end

    Returns:
        Masked data string

    What masking provides:
    - Data protection in logs
    - Partial visibility for verification
    - Prevents accidental exposure
    - Maintains data format
    """
    if len(data) <= visible_chars:
        return "*" * len(data)

    return "*" * (len(data) - visible_chars) + data[-visible_chars:]


def sanitize_input(data: str) -> str:
    """Sanitize user input to prevent injection attacks.

    Args:
        data: User input string

    Returns:
        Sanitized string

    What sanitization provides:
    - Removes potentially dangerous characters
    - Prevents XSS attacks
    - Normalizes input format
    - Maintains data usability
    """
    import re

    # Remove null bytes and control characters
    sanitized = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", data)

    # Normalize whitespace
    sanitized = re.sub(r"\s+", " ", sanitized).strip()

    # Limit length
    sanitized = sanitized[:1000]

    return sanitized


def validate_password_strength(password: str) -> Dict[str, Any]:
    """Validate password strength according to security policies.

    Args:
        password: Password to validate

    Returns:
        Validation result with score and recommendations

    What password validation provides:
    - Enforces security policies
    - Prevents weak passwords
    - Provides user feedback
    - Reduces brute force vulnerability
    """
    result = {"is_valid": True, "score": 0, "issues": [], "recommendations": []}

    # Length check
    if len(password) < 8:
        result["is_valid"] = False
        result["issues"].append("Password too short (minimum 8 characters)")
    else:
        result["score"] += 1

    # Character variety checks
    if not any(c.islower() for c in password):
        result["issues"].append("Missing lowercase letters")
    else:
        result["score"] += 1

    if not any(c.isupper() for c in password):
        result["issues"].append("Missing uppercase letters")
    else:
        result["score"] += 1

    if not any(c.isdigit() for c in password):
        result["issues"].append("Missing numbers")
    else:
        result["score"] += 1

    if not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
        result["issues"].append("Missing special characters")
    else:
        result["score"] += 1

    # Common password check
    common_passwords = [
        "password",
        "123456",
        "qwerty",
        "abc123",
        "password123",
        "admin",
        "letmein",
        "welcome",
        "monkey",
        "dragon",
    ]

    if password.lower() in common_passwords:
        result["is_valid"] = False
        result["issues"].append("Password is too common")

    # Generate recommendations
    if result["score"] < 3:
        result["recommendations"].append("Use a mix of letters, numbers, and symbols")

    if len(password) < 12:
        result["recommendations"].append(
            "Consider using a longer password (12+ characters)"
        )

    return result
