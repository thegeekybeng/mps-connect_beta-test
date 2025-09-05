#!/bin/bash
# MPS Connect Vercel Deployment Script
# Automated deployment to Vercel platform

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VERCEL_TOKEN="${VERCEL_TOKEN:-}"
VERCEL_ORG_ID="${VERCEL_ORG_ID:-}"
VERCEL_PROJECT_ID="${VERCEL_PROJECT_ID:-}"

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
    
    # Check if vercel.json exists
    if [ ! -f "${PROJECT_DIR}/vercel.json" ]; then
        log_error "vercel.json not found. Please create it first."
        exit 1
    fi
    
    # Check if package.json exists
    if [ ! -f "${PROJECT_DIR}/package.json" ]; then
        log_error "package.json not found. Please create it first."
        exit 1
    fi
    
    # Check if Vercel CLI is available
    if ! command -v vercel > /dev/null 2>&1; then
        log_warning "Vercel CLI not found. Installing..."
        install_vercel_cli
    fi
    
    log_success "Prerequisites check completed"
}

# Install Vercel CLI
install_vercel_cli() {
    log_info "Installing Vercel CLI..."
    
    # Install using npm
    if command -v npm > /dev/null 2>&1; then
        npm install -g vercel
    else
        log_error "npm is not installed. Please install Node.js and npm first."
        exit 1
    fi
    
    log_success "Vercel CLI installed successfully"
}

# Authenticate with Vercel
authenticate_vercel() {
    log_info "Authenticating with Vercel..."
    
    if [ -z "$VERCEL_TOKEN" ]; then
        log_error "VERCEL_TOKEN environment variable is not set"
        log_info "Please set your Vercel token:"
        log_info "export VERCEL_TOKEN=your_token_here"
        exit 1
    fi
    
    # Set token
    echo "$VERCEL_TOKEN" | vercel login --token
    
    # Test authentication
    if vercel whoami > /dev/null 2>&1; then
        log_success "Vercel authentication successful"
    else
        log_error "Vercel authentication failed"
        exit 1
    fi
}

# Build frontend
build_frontend() {
    log_info "Building frontend for Vercel..."
    
    cd "$PROJECT_DIR"
    
    # Install dependencies
    log_info "Installing dependencies..."
    npm install
    
    # Build the application
    log_info "Building application..."
    npm run build
    
    # Check if build was successful
    if [ -d "dist" ]; then
        log_success "Frontend build completed successfully"
    else
        log_error "Frontend build failed - dist directory not found"
        exit 1
    fi
}

# Deploy to Vercel
deploy_to_vercel() {
    log_info "Deploying to Vercel..."
    
    cd "$PROJECT_DIR"
    
    # Deploy using Vercel CLI
    if [ -n "$VERCEL_PROJECT_ID" ]; then
        log_info "Deploying to existing project: $VERCEL_PROJECT_ID"
        vercel deploy --prod --token "$VERCEL_TOKEN"
    else
        log_info "Creating new project..."
        vercel deploy --prod --token "$VERCEL_TOKEN" --yes
    fi
    
    log_success "Deployment to Vercel completed"
}

# Configure environment variables
configure_environment() {
    log_info "Configuring environment variables..."
    
    # Set API URL
    if [ -n "$VERCEL_PROJECT_ID" ]; then
        vercel env add VITE_API_URL production --token "$VERCEL_TOKEN"
        vercel env add VITE_APP_NAME production --token "$VERCEL_TOKEN"
        vercel env add VITE_APP_VERSION production --token "$VERCEL_TOKEN"
        vercel env add VITE_ENVIRONMENT production --token "$VERCEL_TOKEN"
        
        log_success "Environment variables configured"
    else
        log_warning "Project ID not found, skipping environment configuration"
    fi
}

# Check deployment status
check_deployment_status() {
    log_info "Checking deployment status..."
    
    # Get deployment URL
    DEPLOYMENT_URL=$(vercel ls --token "$VERCEL_TOKEN" | head -n 2 | tail -n 1 | awk '{print $2}')
    
    if [ -n "$DEPLOYMENT_URL" ]; then
        log_info "Deployment URL: https://$DEPLOYMENT_URL"
        
        # Test deployment
        if curl -f "https://$DEPLOYMENT_URL" > /dev/null 2>&1; then
            log_success "Deployment is accessible"
        else
            log_warning "Deployment accessibility test failed"
        fi
    else
        log_warning "Deployment URL not found"
    fi
}

# Show deployment information
show_deployment_info() {
    log_info "Deployment Information:"
    echo ""
    echo "Frontend deployed to Vercel:"
    vercel ls --token "$VERCEL_TOKEN"
    echo ""
    echo "To view logs:"
    echo "  vercel logs --token $VERCEL_TOKEN"
    echo ""
    echo "To update deployment:"
    echo "  vercel deploy --prod --token $VERCEL_TOKEN"
    echo ""
    echo "To remove deployment:"
    echo "  vercel remove --token $VERCEL_TOKEN"
}

# Cleanup function
cleanup() {
    log_info "Cleaning up..."
    # Add cleanup logic if needed
}

# Main deployment function
main() {
    log_info "Starting MPS Connect Vercel deployment..."
    
    # Pre-deployment checks
    check_prerequisites
    authenticate_vercel
    
    # Build and deploy
    build_frontend
    deploy_to_vercel
    
    # Configure and verify
    configure_environment
    check_deployment_status
    
    # Show information
    show_deployment_info
    
    log_success "Vercel deployment completed successfully!"
    log_info "Your MPS Connect frontend is now running on Vercel"
}

# Handle script arguments
case "${1:-}" in
    "build")
        log_info "Building frontend only..."
        check_prerequisites
        build_frontend
        log_success "Frontend built successfully"
        ;;
    "deploy")
        log_info "Deploying to Vercel..."
        check_prerequisites
        authenticate_vercel
        deploy_to_vercel
        configure_environment
        check_deployment_status
        show_deployment_info
        ;;
    "status")
        log_info "Checking deployment status..."
        check_deployment_status
        ;;
    "logs")
        log_info "Showing deployment logs..."
        vercel logs --token "$VERCEL_TOKEN"
        ;;
    "cleanup")
        cleanup
        ;;
    *)
        echo "Usage: $0 {build|deploy|status|logs|cleanup}"
        echo ""
        echo "Commands:"
        echo "  build     Build frontend only"
        echo "  deploy    Deploy to Vercel platform"
        echo "  status    Check deployment status"
        echo "  logs      Show deployment logs"
        echo "  cleanup   Clean up local resources"
        echo ""
        echo "Environment Variables:"
        echo "  VERCEL_TOKEN      Your Vercel token (required)"
        echo "  VERCEL_ORG_ID     Your Vercel organization ID (optional)"
        echo "  VERCEL_PROJECT_ID Your Vercel project ID (optional)"
        exit 1
        ;;
esac
