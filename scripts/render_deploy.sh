#!/bin/bash
# Render deployment script for MPS Connect
# This script handles pre-deployment tasks like database migrations

set -e

echo "Starting MPS Connect deployment..."

# Wait for database to be ready
echo "Waiting for database connection..."
python -c "
import time
import os
import psycopg2
from urllib.parse import urlparse

def wait_for_db():
    max_retries = 30
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            db_url = os.getenv('DATABASE_URL')
            if not db_url:
                print('DATABASE_URL not set')
                time.sleep(2)
                retry_count += 1
                continue
                
            parsed = urlparse(db_url)
            conn = psycopg2.connect(
                host=parsed.hostname,
                port=parsed.port,
                database=parsed.path[1:],
                user=parsed.username,
                password=parsed.password
            )
            conn.close()
            print('Database connection successful')
            return True
        except Exception as e:
            print(f'Database connection failed: {e}')
            time.sleep(2)
            retry_count += 1
    
    return False

if not wait_for_db():
    print('Failed to connect to database after 60 seconds')
    exit(1)
"

# Run database migrations
echo "Running database migrations..."
alembic upgrade head

# Initialize database if needed
echo "Initializing database..."
python scripts/init_database.py

echo "Deployment preparation complete!"
