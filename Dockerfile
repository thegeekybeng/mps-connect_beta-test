FROM python:3.11-slim
ARG CACHE_BUST=5

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY api/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy application code
COPY api/ /app/api/
COPY database/ /app/database/
COPY security/ /app/security/
COPY governance/ /app/governance/
COPY alembic/ /app/alembic/
COPY scripts/ /app/scripts/
COPY alembic.ini /app/

# Create necessary directories
RUN mkdir -p /app/logs /app/data

# Make scripts executable
RUN chmod +x /app/scripts/*.sh

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose port (Render will set PORT environment variable)
EXPOSE 8000

# Health check uses PORT if valid, defaults to 8000 otherwise
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD sh -c 'PORT_CLEAN=$(echo "${PORT:-8000}" | grep -Eo "[0-9]+" | head -n1); if [ -z "$PORT_CLEAN" ]; then PORT_CLEAN=8000; fi; curl -f http://localhost:${PORT_CLEAN}/healthz || exit 1'

# Start application via Python module that sanitizes PORT internally
CMD ["python", "-m", "api.app"]
