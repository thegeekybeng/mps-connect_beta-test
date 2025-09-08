# MPS Connect Production Deployment Guide

## Overview

This guide previously referenced Render (backend + database) and GitHub Pages (frontend). Those instructions have been removed.

For Google Cloud deployment, use `DEPLOY_GCP.md` at the project root (Cloud Run + Firebase Hosting).

## Prerequisites

- GitHub repository with MPS Connect code
- Cloud account for your chosen provider
- GitHub account with Pages enabled
- Domain name (optional)

## Phase 1: Database Setup (placeholder)

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

### Step 1: Backend + Database Setup (refer to your platform docs)

1. Create or select your hosting account and link your repository as needed.

2. Deploy your database (e.g., Cloud SQL, managed Postgres) and note the connection string.

3. Deploy Backend API using your platform’s standard container instructions and set the required environment variables.

### Step 2: Frontend Setup

1. Deploy the frontend to your hosting provider (e.g., Firebase Hosting) and configure `VITE_API_URL`.

### Step 3: Configuration

1. **Update CORS Settings**

   - Edit `app.py` or set `CORS_ORIGINS` on your hosting platform
   - Include your frontend domain

2. **Test Deployment**
   - Backend health check: `https://<your-api>/healthz`
   - Frontend: `https://<your-frontend-domain>/`

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

Replace with provider-specific cost notes as needed.

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

Refer to your selected platform’s documentation.
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
