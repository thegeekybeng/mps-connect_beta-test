FROM python:3.11-slim
ARG CACHE_BUST=6

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

# Runtime environment
ENV MODEL_DIR=/app/api/artifacts_zs_hier_plus \
    PROVIDERS_JSON=/app/api/providers_map.json \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1

# Expose port (Cloud Run injects $PORT, default 8080)
EXPOSE 8080

# Start command via entrypoint script that sanitizes PORT
CMD ["/app/scripts/start.sh"]
