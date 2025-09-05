#!/bin/bash
# MPS Connect Deployment Script
# Production deployment automation

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ENVIRONMENT="${1:-production}"
COMPOSE_FILE="docker-compose.${ENVIRONMENT}.yml"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is running
check_docker() {
    log_info "Checking Docker availability..."
    if ! docker info > /dev/null 2>&1; then
        log_error "Docker is not running or not accessible"
        exit 1
    fi
    log_success "Docker is available"
}

# Check if Docker Compose is available
check_docker_compose() {
    log_info "Checking Docker Compose availability..."
    if ! command -v docker-compose > /dev/null 2>&1; then
        log_error "Docker Compose is not installed"
        exit 1
    fi
    log_success "Docker Compose is available"
}

# Check environment file
check_environment() {
    log_info "Checking environment configuration..."
    if [ ! -f "${PROJECT_DIR}/.env" ]; then
        if [ -f "${PROJECT_DIR}/env.${ENVIRONMENT}.example" ]; then
            log_warning "Environment file not found, copying from example..."
            cp "${PROJECT_DIR}/env.${ENVIRONMENT}.example" "${PROJECT_DIR}/.env"
            log_warning "Please update .env file with production values before continuing"
            read -p "Press Enter to continue after updating .env file..."
        else
            log_error "Environment file not found and no example available"
            exit 1
        fi
    fi
    log_success "Environment configuration found"
}

# Build Docker images
build_images() {
    log_info "Building Docker images..."
    cd "$PROJECT_DIR"
    
    # Build API image
    log_info "Building API image..."
    docker build -f Dockerfile.api -t mps-connect-api:latest .
    
    # Build Web image
    log_info "Building Web image..."
    docker build -f Dockerfile.web -t mps-connect-web:latest .
    
    log_success "Docker images built successfully"
}

# Deploy services
deploy_services() {
    log_info "Deploying services with ${COMPOSE_FILE}..."
    cd "$PROJECT_DIR"
    
    # Stop existing services
    log_info "Stopping existing services..."
    docker-compose -f "$COMPOSE_FILE" down --remove-orphans
    
    # Start services
    log_info "Starting services..."
    docker-compose -f "$COMPOSE_FILE" up -d
    
    # Wait for services to be healthy
    log_info "Waiting for services to be healthy..."
    sleep 30
    
    # Check service health
    check_service_health
    
    log_success "Services deployed successfully"
}

# Check service health
check_service_health() {
    log_info "Checking service health..."
    
    # Check API service
    if docker-compose -f "$COMPOSE_FILE" ps api | grep -q "Up"; then
        log_success "API service is running"
    else
        log_error "API service is not running"
        docker-compose -f "$COMPOSE_FILE" logs api
        exit 1
    fi
    
    # Check Web service
    if docker-compose -f "$COMPOSE_FILE" ps web | grep -q "Up"; then
        log_success "Web service is running"
    else
        log_error "Web service is not running"
        docker-compose -f "$COMPOSE_FILE" logs web
        exit 1
    fi
    
    # Check Database service
    if docker-compose -f "$COMPOSE_FILE" ps postgres | grep -q "Up"; then
        log_success "Database service is running"
    else
        log_error "Database service is not running"
        docker-compose -f "$COMPOSE_FILE" logs postgres
        exit 1
    fi
    
    # Check Redis service
    if docker-compose -f "$COMPOSE_FILE" ps redis | grep -q "Up"; then
        log_success "Redis service is running"
    else
        log_error "Redis service is not running"
        docker-compose -f "$COMPOSE_FILE" logs redis
        exit 1
    fi
}

# Run database migrations
run_migrations() {
    log_info "Running database migrations..."
    cd "$PROJECT_DIR"
    
    # Run migrations
    docker-compose -f "$COMPOSE_FILE" run --rm migrate
    
    log_success "Database migrations completed"
}

# Show service status
show_status() {
    log_info "Service Status:"
    cd "$PROJECT_DIR"
    docker-compose -f "$COMPOSE_FILE" ps
    
    log_info "Service URLs:"
    echo "  API: http://localhost:8000"
    echo "  Web: http://localhost:3000"
    echo "  Database: localhost:5432"
    echo "  Redis: localhost:6379"
    
    if [ "$ENVIRONMENT" = "production" ]; then
        echo "  Prometheus: http://localhost:9091"
        echo "  Grafana: http://localhost:3001"
    else
        echo "  pgAdmin: http://localhost:5050"
        echo "  RedisInsight: http://localhost:8001"
        echo "  Mailhog: http://localhost:8025"
    fi
}

# Cleanup function
cleanup() {
    log_info "Cleaning up..."
    cd "$PROJECT_DIR"
    docker-compose -f "$COMPOSE_FILE" down --remove-orphans
    log_success "Cleanup completed"
}

# Main deployment function
main() {
    log_info "Starting MPS Connect deployment..."
    log_info "Environment: $ENVIRONMENT"
    log_info "Compose file: $COMPOSE_FILE"
    
    # Pre-deployment checks
    check_docker
    check_docker_compose
    check_environment
    
    # Build and deploy
    build_images
    deploy_services
    run_migrations
    
    # Show final status
    show_status
    
    log_success "Deployment completed successfully!"
    log_info "You can now access the application at the URLs shown above"
}

# Handle script arguments
case "${1:-}" in
    "dev"|"development")
        ENVIRONMENT="dev"
        COMPOSE_FILE="docker-compose.dev.yml"
        main
        ;;
    "prod"|"production")
        ENVIRONMENT="production"
        COMPOSE_FILE="docker-compose.prod.yml"
        main
        ;;
    "stop")
        log_info "Stopping services..."
        cd "$PROJECT_DIR"
        docker-compose -f "$COMPOSE_FILE" down
        log_success "Services stopped"
        ;;
    "logs")
        log_info "Showing service logs..."
        cd "$PROJECT_DIR"
        docker-compose -f "$COMPOSE_FILE" logs -f
        ;;
    "status")
        show_status
        ;;
    "cleanup")
        cleanup
        ;;
    *)
        echo "Usage: $0 {dev|prod|stop|logs|status|cleanup}"
        echo ""
        echo "Commands:"
        echo "  dev        Deploy development environment"
        echo "  prod       Deploy production environment"
        echo "  stop       Stop all services"
        echo "  logs       Show service logs"
        echo "  status     Show service status"
        echo "  cleanup    Stop and remove all containers"
        exit 1
        ;;
esac
