FROM python:3.11-slim
ARG CACHE_BUST=4

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && rm -rf /var/lib/apt/lists/*

COPY api/requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

COPY api/app.py /app/app.py
COPY api/artifacts_zs_hier_plus /app/artifacts_zs_hier_plus
COPY api/providers_map.json /app/providers_map.json

# Copy SSL certs
COPY api/certs /app/certs

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info", "--access-log"]
