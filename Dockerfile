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

# Health check respects PORT if provided (sanitize to integer)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD sh -c 'PORT_CLEAN=${PORT%%.*}; if [ -z "$PORT_CLEAN" ]; then PORT_CLEAN=8000; fi; curl -f http://localhost:${PORT_CLEAN}/healthz || exit 1'

# Start command that respects Render's PORT variable
# Sanitize PORT to be an integer in case it's provided like "10000." or "10000.0"
CMD ["sh", "-c", "PORT_CLEAN=${PORT%%.*}; if [ -z \"$PORT_CLEAN\" ]; then PORT_CLEAN=8000; fi; exec uvicorn api.app:app --host 0.0.0.0 --port ${PORT_CLEAN}"]
