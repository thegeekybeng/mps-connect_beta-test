#!/bin/bash
# MPS Connect Render Deployment Script
# Automated deployment to Render platform

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
RENDER_API_KEY="${RENDER_API_KEY:-}"
RENDER_SERVICE_ID="${RENDER_SERVICE_ID:-}"

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

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if render.yaml exists
    if [ ! -f "${PROJECT_DIR}/render.yaml" ]; then
        log_error "render.yaml not found. Please create it first."
        exit 1
    fi
    
    # Check if Docker is available
    if ! command -v docker > /dev/null 2>&1; then
        log_error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    # Check if Render CLI is available
    if ! command -v render > /dev/null 2>&1; then
        log_warning "Render CLI not found. Installing..."
        install_render_cli
    fi
    
    log_success "Prerequisites check completed"
}

# Install Render CLI
install_render_cli() {
    log_info "Installing Render CLI..."
    
    # Detect OS
    OS=$(uname -s)
    ARCH=$(uname -m)
    
    case $OS in
        Linux*)
            PLATFORM="linux"
            ;;
        Darwin*)
            PLATFORM="darwin"
            ;;
        *)
            log_error "Unsupported operating system: $OS"
            exit 1
            ;;
    esac
    
    # Download and install
    DOWNLOAD_URL="https://github.com/render-oss/cli/releases/latest/download/render-${PLATFORM}-${ARCH}"
    curl -L "$DOWNLOAD_URL" -o /tmp/render
    chmod +x /tmp/render
    sudo mv /tmp/render /usr/local/bin/render
    
    log_success "Render CLI installed successfully"
}

# Authenticate with Render
authenticate_render() {
    log_info "Authenticating with Render..."
    
    if [ -z "$RENDER_API_KEY" ]; then
        log_error "RENDER_API_KEY environment variable is not set"
        log_info "Please set your Render API key:"
        log_info "export RENDER_API_KEY=your_api_key_here"
        exit 1
    fi
    # Login non-interactively using API key, then test authentication
    render login --api-key "$RENDER_API_KEY" > /dev/null 2>&1 || true
    if render auth whoami > /dev/null 2>&1; then
        log_success "Render authentication successful"
    else
        log_error "Render authentication failed"
        exit 1
    fi
}

# Build Docker images
build_images() {
    log_info "Building Docker images for Render..."
    
    cd "$PROJECT_DIR"
    
    # Build API image
    log_info "Building API image..."
    docker build -f Dockerfile.api -t mps-connect-api:latest .
    
    # Build Database image
    log_info "Building Database image..."
    docker build -f Dockerfile.database -t mps-connect-db:latest .
    
    # Build Backup image
    log_info "Building Backup image..."
    docker build -f Dockerfile.backup -t mps-connect-backup:latest .
    
    # Build Monitor image
    log_info "Building Monitor image..."
    docker build -f Dockerfile.monitor -t mps-connect-monitor:latest .
    
    log_success "Docker images built successfully"
}

# Deploy to Render
deploy_to_render() {
    log_info "Deploying to Render..."
    
    cd "$PROJECT_DIR"
    
    # Deploy using render.yaml
    if render services create --file render.yaml; then
        log_success "Services created successfully"
    else
        log_warning "Services may already exist, attempting to update..."
        render services update --file render.yaml
    fi
    
    log_success "Deployment to Render completed"
}

# Check deployment status
check_deployment_status() {
    log_info "Checking deployment status..."
    
    # Wait for services to be ready
    log_info "Waiting for services to be ready..."
    sleep 30
    
    # Check API service
    if render services list | grep -q "mps-connect-api"; then
        API_URL=$(render services list | grep "mps-connect-api" | awk '{print $3}')
        log_info "API service URL: $API_URL"
        
        # Test API health
        if curl -f "$API_URL/healthz" > /dev/null 2>&1; then
            log_success "API service is healthy"
        else
            log_warning "API service health check failed"
        fi
    else
        log_warning "API service not found"
    fi
    
    # Check database service
    if render services list | grep -q "mps-connect-db"; then
        log_success "Database service is running"
    else
        log_warning "Database service not found"
    fi
    
    # Check Redis service
    if render services list | grep -q "mps-connect-redis"; then
        log_success "Redis service is running"
    else
        log_warning "Redis service not found"
    fi
}

# Show deployment information
show_deployment_info() {
    log_info "Deployment Information:"
    echo ""
    echo "Services deployed to Render:"
    render services list
    echo ""
    echo "To view logs:"
    echo "  render logs --service mps-connect-api"
    echo "  render logs --service mps-connect-db"
    echo "  render logs --service mps-connect-redis"
    echo ""
    echo "To update services:"
    echo "  render services update --file render.yaml"
    echo ""
    echo "To delete services:"
    echo "  render services delete --service mps-connect-api"
    echo "  render services delete --service mps-connect-db"
    echo "  render services delete --service mps-connect-redis"
}

# Cleanup function
cleanup() {
    log_info "Cleaning up..."
    # Add cleanup logic if needed
}

# Main deployment function
main() {
    log_info "Starting MPS Connect Render deployment..."
    
    # Pre-deployment checks
    check_prerequisites
    authenticate_render
    
    # Build and deploy
    build_images
    deploy_to_render
    
    # Verify deployment
    check_deployment_status
    
    # Show information
    show_deployment_info
    
    log_success "Render deployment completed successfully!"
    log_info "Your MPS Connect backend is now running on Render"
}

# Handle script arguments
case "${1:-}" in
    "build")
        log_info "Building Docker images only..."
        check_prerequisites
        build_images
        log_success "Docker images built successfully"
        ;;
    "deploy")
        log_info "Deploying to Render..."
        check_prerequisites
        authenticate_render
        deploy_to_render
        check_deployment_status
        show_deployment_info
        ;;
    "status")
        log_info "Checking deployment status..."
        check_deployment_status
        ;;
    "logs")
        SERVICE="${2:-mps-connect-api}"
        log_info "Showing logs for $SERVICE..."
        render logs --service "$SERVICE"
        ;;
    "cleanup")
        cleanup
        ;;
    *)
        echo "Usage: $0 {build|deploy|status|logs|cleanup}"
        echo ""
        echo "Commands:"
        echo "  build     Build Docker images only"
        echo "  deploy    Deploy to Render platform"
        echo "  status    Check deployment status"
        echo "  logs      Show service logs (specify service name)"
        echo "  cleanup   Clean up local resources"
        echo ""
        echo "Environment Variables:"
        echo "  RENDER_API_KEY    Your Render API key (required)"
        echo "  RENDER_SERVICE_ID Your Render service ID (optional)"
        exit 1
        ;;
esac
