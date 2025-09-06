# Render Deployment Configuration

## Web Service Configuration

When creating a new web service on Render, use these settings:

### Basic Configuration

- **Docker Build Context Directory**: `.` (root of repository)
- **Dockerfile Path**: `./Dockerfile`
- **Docker Command**: `python -m api.app`
  *Note*: `api.app` sanitizes Render's auto-generated `PORT` value before starting the server.
- **Pre-Deploy Command**: `bash scripts/render_deploy.sh`
- **Auto-Deploy**: `On Commit`
- **Build Filters**: Include these paths:
  - `api/**`
  - `database/**`
  - `security/**`
  - `governance/**`
  - `alembic/**`
  - `scripts/**`
  - `requirements.txt`
  - `Dockerfile`
  - `alembic.ini`

### Environment Variables

Set these environment variables in Render:

#### Database Configuration

```
DATABASE_URL=postgresql://mps_user:password@localhost:5432/mps_connect
```

#### Security Configuration

```
SECRET_KEY=your-secret-key-here
ENCRYPTION_KEY=your-encryption-key-here
JWT_SECRET_KEY=your-jwt-secret-key-here
```

#### Application Configuration

```
ENVIRONMENT=production
LOG_LEVEL=INFO
CORS_ORIGINS=https://thegeekybeng.github.io
```

#### Model Configuration

```
MODEL_DIR=./api/artifacts_zs_hier_plus
PROVIDERS_JSON=./api/providers_map.json
API_KEY=mps-85-whampoa
```

#### Performance Configuration

```
MAX_WORKERS=4
WORKER_TIMEOUT=30
KEEP_ALIVE_TIMEOUT=5
```

#### Security Configuration

```
RATE_LIMIT_PER_MINUTE=100
RATE_LIMIT_BURST=200
SESSION_TIMEOUT=3600
```

#### Audit Configuration

```
AUDIT_LOG_LEVEL=INFO
AUDIT_LOG_RETENTION_DAYS=730
```

#### Compliance Configuration

```
COMPLIANCE_CHECK_INTERVAL=3600
COMPLIANCE_REPORT_SCHEDULE=0 0 * * 0
```

## Database Setup

1. Create a PostgreSQL database service on Render
2. Use the connection string as `DATABASE_URL`
3. The pre-deploy script will handle migrations

## Deployment Steps

1. **Connect Repository**: Link your GitHub repository to Render
2. **Create Database**: Set up PostgreSQL database service
3. **Create Web Service**: Use the configuration above
4. **Set Environment Variables**: Add all required environment variables
5. **Deploy**: Render will automatically build and deploy

## Health Check

The service includes a health check endpoint at `/healthz` that verifies:

- Database connectivity
- Model loading
- Service status

## Monitoring

- Health checks run every 30 seconds
- Logs are available in Render dashboard
- Metrics are collected for performance monitoring

## Troubleshooting

### Common Issues

1. **Build Failures**

   - Check that all required files are in the repository
   - Verify Dockerfile syntax
   - Check build logs for specific errors

2. **Database Connection Issues**

   - Verify DATABASE_URL is correct
   - Check database service is running
   - Ensure network connectivity

3. **Model Loading Issues**
   - Verify MODEL_DIR path is correct
   - Check that model files are present
   - Verify file permissions

### Support

- Render Documentation: https://docs.render.com
- MPS Connect Issues: Check logs in Render dashboard
