FROM python:3.11-slim
ARG CACHE_BUST=4

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

COPY app.py /app/app.py
COPY artifacts_zs_hier_plus /app/artifacts_zs_hier_plus
COPY providers_map.json /app/providers_map.json

# Copy SSL certs
COPY certs /app/certs
# Force rebuild - updated embeddings (v3)

EXPOSE 8080

# Use provided PORT if set by platform (e.g., Cloud Run)
CMD ["sh", "-c", "PORT_CLEAN=${PORT%%.*}; if [ -z \"$PORT_CLEAN\" ]; then PORT_CLEAN=8080; fi; exec uvicorn app:app --host 0.0.0.0 --port ${PORT_CLEAN}"]
