# MPS Connect Production Deployment Guide

## Overview

This guide covers deploying MPS Connect to Render (backend + database) and Vercel (frontend) for production demo.

For detailed Render configuration, see [RENDER_DEPLOYMENT.md](./RENDER_DEPLOYMENT.md).

## Prerequisites

- GitHub repository with MPS Connect code
- Render account (free tier)
- Vercel account (free tier)
- Domain name (optional)

## Phase 1: Database Setup (PostgreSQL on Render)

### What PostgreSQL Does Functionally

PostgreSQL is a powerful relational database that provides:

- **ACID Compliance**: Ensures data integrity with Atomicity, Consistency, Isolation, Durability
- **SQL Standard Compliance**: Full SQL support with advanced features
- **Data Types**: Rich set including JSON, arrays, custom types
- **Indexing**: Multiple index types for performance optimization
- **Concurrency**: Multi-version concurrency control (MVCC)
- **Extensions**: Custom functions, data types, and operators

### Suitable Project Types

- Web applications (e-commerce, CMS, user management)
- Analytics and reporting systems
- Financial and banking applications
- Government systems with audit requirements
- AI/ML feature stores and metadata

### Limitations

- Higher memory usage than MySQL
- Steeper learning curve for advanced features
- Write performance can be slower than NoSQL
- Limited horizontal scaling compared to distributed databases

### Alternatives

- **MySQL**: Lighter, faster for simple queries
- **MongoDB**: Document-based, better for unstructured data
- **Redis**: In-memory, excellent for caching
- **SQLite**: Embedded, single-user applications

## Deployment Steps

### Step 1: Render Backend + Database Setup

1. **Create Render Account**

   - Go to [render.com](https://render.com)
   - Sign up with GitHub account
   - Connect your repository

2. **Deploy PostgreSQL Database**

   - Create new PostgreSQL service
   - Choose "Free" plan
   - Select Singapore region
   - Database name: `mpsconnect`
   - User: `mpsconnect`
   - Note the connection string

3. **Deploy Backend API**
   - Create new Web Service
   - Connect GitHub repository
   - Choose "Free" plan
   - **Docker Build Context Directory**: `.` (root of repository)
   - **Dockerfile Path**: `./Dockerfile`
   - **Docker Command**: `uvicorn api.app:app --host 0.0.0.0 --port $PORT`
   - **Pre-Deploy Command**: `bash scripts/render_deploy.sh`
   - **Auto-Deploy**: `On Commit`
   - **Build Filters**: Include paths: `api/**`, `database/**`, `security/**`, `governance/**`, `alembic/**`, `scripts/**`, `requirements.txt`, `Dockerfile`, `alembic.ini`
   - Environment Variables:
     - `DATABASE_URL`: (from PostgreSQL service)
     - `MODEL_DIR`: `./api/artifacts_zs_hier_plus`
     - `PROVIDERS_JSON`: `./api/providers_map.json`
     - `API_KEY`: `mps-85-whampoa`
     - `SECRET_KEY`: (generate random string)
     - `ENCRYPTION_KEY`: (generate random string)
     - `JWT_SECRET_KEY`: (generate random string)
     - `ENVIRONMENT`: `production`
     - `LOG_LEVEL`: `INFO`
     - `CORS_ORIGINS`: `https://your-frontend.vercel.app`

### Step 2: Vercel Frontend Setup

1. **Create Vercel Account**

   - Go to [vercel.com](https://vercel.com)
   - Sign up with GitHub account
   - Connect your repository

2. **Deploy Frontend**
   - Import project from GitHub
   - Framework: Other
   - Build Command: (leave empty)
   - Output Directory: (leave empty)
   - Environment Variables:
     - `API_BASE_URL`: (your Render API URL)

### Step 3: Configuration

1. **Update CORS Settings**

   - Edit `api/app.py`
   - Update `ALLOWED_ORIGINS` with Vercel URL

2. **Test Deployment**
   - Backend health check: `https://your-api.onrender.com/healthz`
   - Frontend: `https://your-app.vercel.app`

## Database Schema

The database includes these core tables:

- **users**: User authentication and roles
- **cases**: Constituent cases and status
- **conversations**: Chat interactions and extracted facts
- **letters**: Generated letters and approval status
- **audit_logs**: Immutable audit trail
- **data_lineage**: Data relationship tracking
- **user_activities**: User action logging
- **sessions**: Authentication sessions
- **permissions**: Role-based access control
- **access_logs**: API request tracking

## Security Features

- **Encryption**: TLS 1.3 for all connections
- **Authentication**: JWT-based session management
- **Authorization**: Role-based access control
- **Audit Logging**: Immutable action records
- **Data Retention**: Configurable cleanup policies
- **Input Validation**: XSS and injection protection

## Monitoring and Health Checks

- **Health Endpoint**: `/healthz`
- **Database Status**: Connection and table counts
- **Performance Metrics**: Response times and error rates
- **Audit Reports**: Automated compliance reporting

## Cost Analysis

**Render (Free Tier)**

- Backend: $0/month (750 hours)
- Database: $0/month (1GB storage)
- Bandwidth: $0/month (100GB)

**Vercel (Free Tier)**

- Frontend: $0/month
- Bandwidth: $0/month (100GB)
- Functions: $0/month (100GB-hours)

**Total Cost: $0/month**

## Troubleshooting

### Common Issues

1. **Database Connection Failed**

   - Check DATABASE_URL environment variable
   - Verify PostgreSQL service is running
   - Check network connectivity

2. **CORS Errors**

   - Update ALLOWED_ORIGINS in api/app.py
   - Verify frontend URL is correct

3. **Build Failures**
   - Check requirements.txt dependencies
   - Verify Python version compatibility
   - Check build logs for specific errors

### Support

- Render: [docs.render.com](https://docs.render.com)
- Vercel: [vercel.com/docs](https://vercel.com/docs)
- PostgreSQL: [postgresql.org/docs](https://postgresql.org/docs)

## Next Steps

After successful deployment:

1. **Phase 2**: Implement security module with JWT authentication
2. **Phase 3**: Add governance module with audit logging
3. **Phase 4**: Optimize Docker deployment
4. **Phase 5**: Set up monitoring and alerting

## Verification Checklist

- [ ] Database tables created successfully
- [ ] Backend API responding to health checks
- [ ] Frontend loading without errors
- [ ] CORS configuration working
- [ ] Environment variables set correctly
- [ ] SSL certificates active
- [ ] Audit logging functional
- [ ] Data retention policies active
