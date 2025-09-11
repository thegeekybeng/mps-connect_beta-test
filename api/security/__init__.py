"""Security module bootstrap for MPS Connect (API namespace).

Provides graceful fallbacks when optional submodules are not present.
"""

from typing import Any

try:
    from .security_manager import SecurityManager  # type: ignore
except Exception:  # pragma: no cover

    class SecurityManager:  # type: ignore
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass


try:
    from .encryption import EncryptionManager  # type: ignore
except Exception:  # pragma: no cover

    class EncryptionManager:  # type: ignore
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass


try:
    from .jwt_handler import JWTHandler  # type: ignore
except Exception:  # pragma: no cover

    class JWTHandler:  # type: ignore
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass


try:
    from .key_manager import KeyManager  # type: ignore
except Exception:  # pragma: no cover

    class KeyManager:  # type: ignore
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass


__all__ = ["SecurityManager", "EncryptionManager", "JWTHandler", "KeyManager"]
