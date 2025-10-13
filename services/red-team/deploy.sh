#!/bin/bash

# AI Red Team Service Deployment Script
# Comprehensive deployment automation for the red team service

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SERVICE_NAME="ai-red-team-service"
VERSION=${VERSION:-"latest"}
ENVIRONMENT=${ENVIRONMENT:-"development"}
DOCKER_REGISTRY=${DOCKER_REGISTRY:-"localhost:5000"}
NAMESPACE=${NAMESPACE:-"red-team"}

# Functions
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

check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check if Docker Compose is installed
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    # Check if kubectl is installed (for Kubernetes deployment)
    if ! command -v kubectl &> /dev/null; then
        log_warning "kubectl is not installed. Kubernetes deployment will be skipped."
    fi
    
    log_success "Prerequisites check completed"
}

build_images() {
    log_info "Building Docker images..."
    
    # Build main service image
    log_info "Building main service image..."
    docker build -t ${DOCKER_REGISTRY}/${SERVICE_NAME}:${VERSION} .
    
    # Build attack data generator image
    log_info "Building attack data generator image..."
    docker build -f Dockerfile.attack-data-generator -t ${DOCKER_REGISTRY}/${SERVICE_NAME}-data-generator:${VERSION} .
    
    log_success "Docker images built successfully"
}

push_images() {
    if [ "$ENVIRONMENT" != "local" ]; then
        log_info "Pushing images to registry..."
        
        docker push ${DOCKER_REGISTRY}/${SERVICE_NAME}:${VERSION}
        docker push ${DOCKER_REGISTRY}/${SERVICE_NAME}-data-generator:${VERSION}
        
        log_success "Images pushed to registry"
    else
        log_info "Skipping image push for local environment"
    fi
}

deploy_docker_compose() {
    log_info "Deploying with Docker Compose..."
    
    # Create environment-specific compose file
    if [ -f "docker-compose.${ENVIRONMENT}.yml" ]; then
        COMPOSE_FILE="docker-compose.${ENVIRONMENT}.yml"
    else
        COMPOSE_FILE="docker-compose.yml"
    fi
    
    # Deploy services
    docker-compose -f ${COMPOSE_FILE} up -d
    
    # Wait for services to be ready
    log_info "Waiting for services to be ready..."
    sleep 30
    
    # Check service health
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        log_success "Main service is healthy"
    else
        log_error "Main service health check failed"
        exit 1
    fi
    
    log_success "Docker Compose deployment completed"
}

deploy_kubernetes() {
    if command -v kubectl &> /dev/null; then
        log_info "Deploying to Kubernetes..."
        
        # Create namespace if it doesn't exist
        kubectl create namespace ${NAMESPACE} --dry-run=client -o yaml | kubectl apply -f -
        
        # Apply Kubernetes manifests
        if [ -d "k8s" ]; then
            kubectl apply -f k8s/ -n ${NAMESPACE}
        else
            log_warning "Kubernetes manifests not found. Skipping Kubernetes deployment."
        fi
        
        log_success "Kubernetes deployment completed"
    else
        log_warning "kubectl not found. Skipping Kubernetes deployment."
    fi
}

run_tests() {
    log_info "Running deployment tests..."
    
    # Test main service
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        log_success "Main service health check passed"
    else
        log_error "Main service health check failed"
        exit 1
    fi
    
    # Test attack data generator service
    if curl -f http://localhost:8001/health > /dev/null 2>&1; then
        log_success "Attack data generator service health check passed"
    else
        log_warning "Attack data generator service health check failed"
    fi
    
    # Run basic API tests
    log_info "Running basic API tests..."
    python3 -m pytest tests/integration/ -v --tb=short
    
    log_success "Deployment tests completed"
}

setup_monitoring() {
    log_info "Setting up monitoring..."
    
    # Start Prometheus and Grafana if not already running
    if [ -d "monitoring" ]; then
        cd monitoring
        docker-compose up -d prometheus grafana
        cd ..
        log_success "Monitoring stack started"
    else
        log_warning "Monitoring configuration not found"
    fi
}

cleanup() {
    log_info "Cleaning up..."
    
    # Stop and remove containers
    docker-compose down
    
    # Remove unused images
    docker image prune -f
    
    log_success "Cleanup completed"
}

show_status() {
    log_info "Service Status:"
    
    # Docker Compose status
    docker-compose ps
    
    # Service health
    echo ""
    log_info "Health Checks:"
    curl -s http://localhost:8000/health | jq . || echo "Main service not responding"
    curl -s http://localhost:8001/health | jq . || echo "Attack data generator not responding"
    
    # Resource usage
    echo ""
    log_info "Resource Usage:"
    docker stats --no-stream
}

# Main deployment function
deploy() {
    log_info "Starting deployment of AI Red Team Service..."
    log_info "Environment: ${ENVIRONMENT}"
    log_info "Version: ${VERSION}"
    log_info "Registry: ${DOCKER_REGISTRY}"
    
    check_prerequisites
    build_images
    push_images
    deploy_docker_compose
    deploy_kubernetes
    setup_monitoring
    run_tests
    
    log_success "Deployment completed successfully!"
    show_status
}

# Command line interface
case "${1:-deploy}" in
    "deploy")
        deploy
        ;;
    "build")
        check_prerequisites
        build_images
        ;;
    "push")
        push_images
        ;;
    "docker-compose")
        deploy_docker_compose
        ;;
    "kubernetes")
        deploy_kubernetes
        ;;
    "test")
        run_tests
        ;;
    "monitoring")
        setup_monitoring
        ;;
    "status")
        show_status
        ;;
    "cleanup")
        cleanup
        ;;
    "help")
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  deploy        - Full deployment (default)"
        echo "  build         - Build Docker images only"
        echo "  push          - Push images to registry"
        echo "  docker-compose - Deploy with Docker Compose only"
        echo "  kubernetes    - Deploy to Kubernetes only"
        echo "  test          - Run deployment tests"
        echo "  monitoring    - Setup monitoring stack"
        echo "  status        - Show service status"
        echo "  cleanup       - Clean up resources"
        echo "  help          - Show this help message"
        echo ""
        echo "Environment variables:"
        echo "  VERSION       - Image version (default: latest)"
        echo "  ENVIRONMENT   - Environment (default: development)"
        echo "  DOCKER_REGISTRY - Docker registry (default: localhost:5000)"
        echo "  NAMESPACE     - Kubernetes namespace (default: red-team)"
        ;;
    *)
        log_error "Unknown command: $1"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac
