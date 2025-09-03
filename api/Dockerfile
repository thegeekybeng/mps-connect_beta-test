FROM python:3.11-slim
ARG CACHE_BUST=3

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

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]