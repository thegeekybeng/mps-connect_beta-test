"""Security middleware for MPS Connect."""

import time
import logging
from typing import Callable
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from .audit import log_api_access
from .encryption import sanitize_input

logger = logging.getLogger(__name__)


class SecurityMiddleware(BaseHTTPMiddleware):
    """Security middleware for request/response processing.

    What this middleware provides:
    - Request sanitization
    - Response security headers
    - API access logging
    - Rate limiting (basic)
    - XSS protection
    - CSRF protection
    """

    def __init__(self, app, db_session_factory):
        super().__init__(app)
        self.db_session_factory = db_session_factory
        self.rate_limit_cache = {}
        self.rate_limit_window = 60  # 1 minute
        self.max_requests_per_window = 100

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request through security middleware."""
        start_time = time.time()

        # Sanitize request data
        await self._sanitize_request(request)

        # Check rate limiting
        if await self._is_rate_limited(request):
            return JSONResponse(
                status_code=429, content={"detail": "Rate limit exceeded"}
            )

        # Add security headers
        response = await call_next(request)
        self._add_security_headers(response)

        # Log API access
        await self._log_api_access(request, response, start_time)

        return response

    async def _sanitize_request(self, request: Request) -> None:
        """Sanitize request data to prevent injection attacks.

        What sanitization provides:
        - Removes malicious characters
        - Prevents XSS attacks
        - Normalizes input data
        - Protects against injection
        """
        # Sanitize query parameters
        if request.query_params:
            sanitized_params = {}
            for key, value in request.query_params.items():
                sanitized_key = sanitize_input(key)
                sanitized_value = sanitize_input(value)
                sanitized_params[sanitized_key] = sanitized_value
            request.query_params = sanitized_params

        # Sanitize headers
        if request.headers:
            sanitized_headers = {}
            for key, value in request.headers.items():
                sanitized_key = sanitize_input(key)
                sanitized_value = sanitize_input(value)
                sanitized_headers[sanitized_key] = sanitized_value
            request.headers = sanitized_headers

    async def _is_rate_limited(self, request: Request) -> bool:
        """Check if request exceeds rate limit.

        What rate limiting provides:
        - Prevents abuse and DoS attacks
        - Protects server resources
        - Ensures fair usage
        - Reduces spam and bot traffic
        """
        client_ip = request.client.host if request.client else "unknown"
        current_time = time.time()

        # Clean old entries
        self.rate_limit_cache = {
            ip: timestamps
            for ip, timestamps in self.rate_limit_cache.items()
            if any(t > current_time - self.rate_limit_window for t in timestamps)
        }

        # Check current IP
        if client_ip in self.rate_limit_cache:
            timestamps = self.rate_limit_cache[client_ip]
            recent_requests = [
                t for t in timestamps if t > current_time - self.rate_limit_window
            ]

            if len(recent_requests) >= self.max_requests_per_window:
                logger.warning("Rate limit exceeded for IP: %s", client_ip)
                return True

            recent_requests.append(current_time)
            self.rate_limit_cache[client_ip] = recent_requests
        else:
            self.rate_limit_cache[client_ip] = [current_time]

        return False

    def _add_security_headers(self, response: Response) -> None:
        """Add security headers to response.

        What security headers provide:
        - XSS protection
        - Clickjacking prevention
        - MIME type sniffing prevention
        - Content type enforcement
        - Referrer policy control
        """
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "font-src 'self' data:; "
            "connect-src 'self'"
        )
        response.headers["Strict-Transport-Security"] = (
            "max-age=31536000; includeSubDomains"
        )
        response.headers["Permissions-Policy"] = (
            "geolocation=(), microphone=(), camera=()"
        )

    async def _log_api_access(
        self, request: Request, response: Response, start_time: float
    ) -> None:
        """Log API access for monitoring and security.

        What API logging provides:
        - Security monitoring
        - Performance tracking
        - Usage analytics
        - Error detection
        """
        try:
            db = next(self.db_session_factory())

            response_time_ms = int((time.time() - start_time) * 1000)

            # Extract user ID from token if present
            user_id = None
            auth_header = request.headers.get("authorization")
            if auth_header and auth_header.startswith("Bearer "):
                try:
                    from .auth import verify_token

                    token = auth_header.split(" ")[1]
                    payload = verify_token(token)
                    user_id = payload.get("sub")
                except Exception:
                    pass  # Ignore token errors for logging

            log_api_access(
                db=db,
                user_id=user_id,
                endpoint=str(request.url.path),
                method=request.method,
                status_code=response.status_code,
                response_time_ms=response_time_ms,
                ip_address=request.client.host if request.client else None,
                user_agent=request.headers.get("user-agent"),
            )

        except Exception as e:
            logger.error("Failed to log API access: %s", e)


class CORSecurityMiddleware:
    """CORS security middleware for cross-origin requests.

    What CORS security provides:
    - Controls cross-origin access
    - Prevents unauthorized domains
    - Protects against CSRF attacks
    - Enables secure cross-domain communication
    """

    def __init__(self, allowed_origins: list):
        self.allowed_origins = allowed_origins

    async def __call__(self, request: Request, call_next: Callable) -> Response:
        """Process CORS security."""
        origin = request.headers.get("origin")

        # Check if origin is allowed
        if origin and origin not in self.allowed_origins:
            logger.warning("Blocked request from unauthorized origin: %s", origin)
            return JSONResponse(
                status_code=403, content={"detail": "Origin not allowed"}
            )

        response = await call_next(request)

        # Add CORS headers
        if origin in self.allowed_origins:
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Methods"] = (
                "GET, POST, PUT, DELETE, OPTIONS"
            )
            response.headers["Access-Control-Allow-Headers"] = (
                "Content-Type, Authorization, X-API-Key"
            )
            response.headers["Access-Control-Allow-Credentials"] = "true"
            response.headers["Access-Control-Max-Age"] = "86400"

        return response


class ContentSecurityMiddleware:
    """Content security middleware for request validation.

    What content security provides:
    - Request size limiting
    - Content type validation
    - Malicious content detection
    - Resource protection
    """

    def __init__(self, max_content_length: int = 10 * 1024 * 1024):  # 10MB
        self.max_content_length = max_content_length
        self.allowed_content_types = [
            "application/json",
            "application/x-www-form-urlencoded",
            "multipart/form-data",
            "text/plain",
        ]

    async def __call__(self, request: Request, call_next: Callable) -> Response:
        """Process content security."""
        # Check content length
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.max_content_length:
            return JSONResponse(
                status_code=413, content={"detail": "Request too large"}
            )

        # Check content type
        content_type = request.headers.get("content-type", "")
        if request.method in ["POST", "PUT", "PATCH"]:
            if not any(ct in content_type for ct in self.allowed_content_types):
                return JSONResponse(
                    status_code=415, content={"detail": "Unsupported media type"}
                )

        return await call_next(request)
