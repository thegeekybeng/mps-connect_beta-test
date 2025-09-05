# MPS Connect Docker Deployment Guide

## Overview

This guide explains the comprehensive Docker containerization implementation for MPS Connect, covering development and production environments with monitoring and security features.

## Docker Containerization

### **What Docker Containerization Does Functionally:**

- **Application Packaging**: Bundles application with all dependencies
- **Environment Isolation**: Consistent runtime environment across systems
- **Deployment Portability**: Run anywhere Docker is supported
- **Resource Management**: Controlled resource allocation and limits
- **Scalability**: Easy horizontal and vertical scaling
- **Version Control**: Immutable application versions

**Suitable Project Types:**

- Microservices architectures
- Cloud-native applications
- Multi-environment deployments
- CI/CD pipelines
- Development and production environments

**Limitations:**

- Learning curve for Docker concepts
- Container orchestration complexity
- Storage and networking considerations
- Security configuration requirements

**Alternatives:**

- **Virtual Machines**: Full OS virtualization
- **Serverless**: Function-as-a-Service deployment
- **Bare Metal**: Direct hardware deployment
- **PaaS**: Platform-as-a-Service solutions

## Architecture Overview

### **Multi-Service Architecture**

**Core Services:**

1. **API Service**: FastAPI backend with security and governance
2. **Web Service**: React frontend with Nginx
3. **Database**: PostgreSQL with immutable audit logs
4. **Cache**: Redis for session storage and caching
5. **Proxy**: Nginx reverse proxy with security hardening

**Development Services:**

- **pgAdmin**: Database administration
- **RedisInsight**: Redis management
- **Mailhog**: Email testing
- **Hot Reloading**: Development with live updates

**Production Services:**

- **Prometheus**: Metrics collection and monitoring
- **Grafana**: Visualization and alerting
- **SSL/TLS**: HTTPS encryption
- **Load Balancing**: High availability

### **Container Security**

**Security Features:**

- **Non-root Users**: All containers run as non-root
- **Read-only Filesystems**: Immutable container filesystems
- **Security Headers**: Comprehensive HTTP security headers
- **Rate Limiting**: DDoS protection and abuse prevention
- **Network Isolation**: Private container networks
- **Resource Limits**: Memory and CPU constraints

## Docker Files

### **API Dockerfile (`Dockerfile.api`)**

**What the API Dockerfile Provides:**

- **Multi-stage Build**: Optimized production image
- **Security Hardening**: Non-root user, minimal attack surface
- **Performance Optimization**: Gunicorn with multiple workers
- **Health Checks**: Automatic service monitoring
- **Resource Management**: Memory and CPU limits

**Key Features:**

```dockerfile
# Multi-stage build for optimization
FROM python:3.11-slim as builder
# Install build dependencies
# Install Python packages

FROM python:3.11-slim as production
# Runtime dependencies only
# Non-root user for security
# Health checks for monitoring
```

### **Web Dockerfile (`Dockerfile.web`)**

**What the Web Dockerfile Provides:**

- **React Build**: Optimized production build
- **Nginx Serving**: High-performance static file serving
- **Security Headers**: HTTP security configuration
- **Compression**: Gzip compression for performance
- **Caching**: Static asset caching

**Key Features:**

```dockerfile
# Build stage with Node.js
FROM node:18-alpine as builder
# Install dependencies and build

# Production stage with Nginx
FROM nginx:alpine as production
# Copy built assets
# Configure Nginx
```

## Docker Compose Configurations

### **Production Configuration (`docker-compose.prod.yml`)**

**What Production Compose Provides:**

- **High Availability**: Multiple service instances
- **Security Hardening**: Production-grade security
- **Monitoring**: Prometheus and Grafana
- **SSL/TLS**: HTTPS encryption
- **Backup**: Automated database backups
- **Scaling**: Resource limits and scaling

**Services Included:**

- **PostgreSQL**: Database with audit logging
- **Redis**: Session storage and caching
- **API**: FastAPI backend service
- **Web**: React frontend service
- **Nginx**: Reverse proxy and load balancer
- **Prometheus**: Metrics collection
- **Grafana**: Monitoring dashboard

### **Development Configuration (`docker-compose.dev.yml`)**

**What Development Compose Provides:**

- **Hot Reloading**: Live code updates
- **Debug Tools**: Development utilities
- **Database Admin**: pgAdmin interface
- **Cache Admin**: RedisInsight interface
- **Email Testing**: Mailhog for email testing
- **Easy Debugging**: Development-friendly configuration

**Services Included:**

- **PostgreSQL**: Development database
- **Redis**: Development cache
- **API**: Development API with hot reload
- **Web**: Development frontend with hot reload
- **pgAdmin**: Database administration
- **RedisInsight**: Redis management
- **Mailhog**: Email testing

## Nginx Configuration

### **Production Nginx (`infra/nginx/prod.conf`)**

**What Production Nginx Provides:**

- **Reverse Proxy**: Route requests to appropriate services
- **Load Balancing**: Distribute load across instances
- **SSL/TLS**: HTTPS encryption and security
- **Security Headers**: Comprehensive HTTP security
- **Rate Limiting**: DDoS protection
- **Compression**: Gzip compression for performance
- **Caching**: Static asset caching

**Security Features:**

```nginx
# Security headers
add_header X-Frame-Options "DENY" always;
add_header X-Content-Type-Options "nosniff" always;
add_header X-XSS-Protection "1; mode=block" always;
add_header Strict-Transport-Security "max-age=31536000" always;

# Rate limiting
limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
limit_req zone=api burst=20 nodelay;
```

## Monitoring and Observability

### **Prometheus Configuration**

**What Prometheus Provides:**

- **Metrics Collection**: System and application metrics
- **Time Series Data**: Historical performance data
- **Alerting**: Automated alert generation
- **Service Discovery**: Automatic service detection
- **Data Retention**: Configurable data retention policies

**Monitored Services:**

- **API Service**: Request rates, response times, errors
- **Database**: Connection counts, query performance
- **Redis**: Cache hit rates, memory usage
- **Nginx**: Request rates, response times
- **System**: CPU, memory, disk usage

### **Grafana Dashboards**

**What Grafana Provides:**

- **Visualization**: Interactive dashboards and graphs
- **Alerting**: Visual alert management
- **Data Exploration**: Interactive data analysis
- **Custom Dashboards**: Tailored monitoring views
- **Team Collaboration**: Shared monitoring views

**Dashboard Features:**

- **System Overview**: High-level system health
- **API Performance**: Request rates and response times
- **Database Metrics**: Connection and query performance
- **Error Tracking**: Error rates and types
- **Resource Usage**: CPU, memory, and disk usage

## Deployment Scripts

### **Deployment Script (`scripts/deploy.sh`)**

**What the Deployment Script Provides:**

- **Automated Deployment**: One-command deployment
- **Environment Management**: Development and production environments
- **Health Checks**: Automatic service health verification
- **Database Migrations**: Automated schema updates
- **Service Management**: Start, stop, and restart services
- **Logging**: Comprehensive deployment logging

**Usage:**

```bash
# Development deployment
./scripts/deploy.sh dev

# Production deployment
./scripts/deploy.sh prod

# Stop services
./scripts/deploy.sh stop

# View logs
./scripts/deploy.sh logs

# Check status
./scripts/deploy.sh status
```

## Environment Configuration

### **Environment Variables**

**Database Configuration:**

```bash
POSTGRES_DB=mps_connect
POSTGRES_USER=mps_user
POSTGRES_PASSWORD=secure_password
```

**Security Configuration:**

```bash
SECRET_KEY=your_secret_key
ENCRYPTION_KEY=your_encryption_key
JWT_SECRET_KEY=your_jwt_secret_key
```

**Application Configuration:**

```bash
ENVIRONMENT=production
LOG_LEVEL=INFO
CORS_ORIGINS=https://your-domain.com
```

**Monitoring Configuration:**

```bash
GRAFANA_PASSWORD=secure_grafana_password
PROMETHEUS_RETENTION=30d
```

## Security Features

### **Container Security**

**Security Measures:**

- **Non-root Users**: All containers run as non-root
- **Read-only Filesystems**: Immutable container filesystems
- **Security Headers**: Comprehensive HTTP security
- **Network Isolation**: Private container networks
- **Resource Limits**: Memory and CPU constraints
- **Health Checks**: Automatic service monitoring

**Security Headers:**

```nginx
X-Frame-Options: DENY
X-Content-Type-Options: nosniff
X-XSS-Protection: 1; mode=block
Strict-Transport-Security: max-age=31536000
Content-Security-Policy: default-src 'self'
```

### **Network Security**

**Network Configuration:**

- **Private Networks**: Isolated container networks
- **Port Mapping**: Controlled port exposure
- **SSL/TLS**: Encrypted communication
- **Rate Limiting**: DDoS protection
- **Access Control**: Role-based access control

## Performance Optimization

### **Resource Management**

**Resource Limits:**

```yaml
deploy:
  resources:
    limits:
      memory: 1G
      cpus: "0.5"
    reservations:
      memory: 512M
      cpus: "0.25"
```

**Performance Features:**

- **Multi-stage Builds**: Optimized image sizes
- **Layer Caching**: Faster builds
- **Health Checks**: Automatic service monitoring
- **Load Balancing**: Request distribution
- **Caching**: Static asset caching
- **Compression**: Gzip compression

### **Database Optimization**

**PostgreSQL Configuration:**

- **Connection Pooling**: Efficient connection management
- **Query Optimization**: Indexed queries
- **Backup Strategy**: Automated backups
- **Monitoring**: Performance metrics
- **Audit Logging**: Immutable audit trail

## Development Workflow

### **Development Environment**

**Features:**

- **Hot Reloading**: Live code updates
- **Debug Tools**: Development utilities
- **Database Admin**: pgAdmin interface
- **Cache Admin**: RedisInsight interface
- **Email Testing**: Mailhog for email testing

**Usage:**

```bash
# Start development environment
./scripts/deploy.sh dev

# Access services
# API: http://localhost:8000
# Web: http://localhost:3000
# pgAdmin: http://localhost:5050
# RedisInsight: http://localhost:8001
```

### **Production Workflow**

**Features:**

- **High Availability**: Multiple service instances
- **Monitoring**: Prometheus and Grafana
- **SSL/TLS**: HTTPS encryption
- **Backup**: Automated database backups
- **Scaling**: Resource limits and scaling

**Usage:**

```bash
# Deploy production environment
./scripts/deploy.sh prod

# Access services
# API: https://your-domain.com/api
# Web: https://your-domain.com
# Prometheus: http://localhost:9091
# Grafana: http://localhost:3001
```

## Troubleshooting

### **Common Issues**

**1. Container Startup Issues**

- **Symptoms**: Containers fail to start
- **Causes**: Missing environment variables, port conflicts
- **Solutions**: Check .env file, verify port availability

**2. Database Connection Issues**

- **Symptoms**: API cannot connect to database
- **Causes**: Database not ready, wrong credentials
- **Solutions**: Wait for database startup, check credentials

**3. Service Health Issues**

- **Symptoms**: Health checks failing
- **Causes**: Service not ready, configuration errors
- **Solutions**: Check service logs, verify configuration

**4. Performance Issues**

- **Symptoms**: Slow response times
- **Causes**: Resource limits, inefficient queries
- **Solutions**: Increase resource limits, optimize queries

### **Debugging Commands**

**View Service Logs:**

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f api

# Last 100 lines
docker-compose logs --tail=100 api
```

**Check Service Status:**

```bash
# Service status
docker-compose ps

# Service health
docker-compose exec api curl http://localhost:8000/health
```

**Access Service Shells:**

```bash
# API service shell
docker-compose exec api bash

# Database shell
docker-compose exec postgres psql -U mps_user -d mps_connect
```

## Best Practices

### **Development**

1. **Use Development Environment**: Always use dev compose for development
2. **Environment Variables**: Never commit .env files
3. **Hot Reloading**: Use development mode for faster iteration
4. **Debug Tools**: Use provided admin interfaces
5. **Testing**: Test in development before production

### **Production**

1. **Security**: Use strong passwords and secrets
2. **Monitoring**: Set up proper monitoring and alerting
3. **Backups**: Implement automated backup strategy
4. **Updates**: Regular security updates
5. **Scaling**: Monitor resource usage and scale as needed

### **Maintenance**

1. **Log Rotation**: Implement log rotation policies
2. **Resource Monitoring**: Monitor resource usage
3. **Security Updates**: Regular security updates
4. **Backup Verification**: Regular backup testing
5. **Performance Tuning**: Regular performance optimization

## Conclusion

The Docker deployment implementation provides:

- **Complete Containerization**: All services containerized
- **Environment Isolation**: Separate dev and production environments
- **Security Hardening**: Production-grade security features
- **Monitoring**: Comprehensive monitoring and alerting
- **Automation**: Automated deployment and management
- **Scalability**: Easy horizontal and vertical scaling

This Docker architecture ensures that MPS Connect can be deployed consistently across different environments while maintaining security, performance, and reliability standards required for government use.
