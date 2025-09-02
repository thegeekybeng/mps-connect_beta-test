FROM python:3.11-slim
WORKDIR /app

# Copy requirements
COPY api/requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# Copy app.py
COPY api/app.py /app/app.py

# Copy artifacts + providers
COPY api/artifacts_zs_hier_plus /app/artifacts_zs_hier_plus
COPY api/providers_map.json /app/providers_map.json

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
