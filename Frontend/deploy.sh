#!/bin/bash

# Cosmic Analysts ExoAI Deployment Script
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="cosmic-analysts-exoai"
DOCKER_REGISTRY=""
VERSION="latest"

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

# Check if Docker is installed
check_docker() {
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    log_success "Docker and Docker Compose are available"
}

# Create necessary directories
setup_directories() {
    log_info "Setting up directories..."
    mkdir -p models exports extracted_modules benchmarks
    log_success "Directories created"
}

# Build Docker images
build_images() {
    log_info "Building Docker images..."
    
    # Build main image
    docker build -t ${PROJECT_NAME}:${VERSION} .
    log_success "Main image built"
    
    # Build development image
    docker build -t ${PROJECT_NAME}-dev:${VERSION} --target development .
    log_success "Development image built"
    
    # Build production image
    docker build -t ${PROJECT_NAME}-prod:${VERSION} --target production .
    log_success "Production image built"
}

# Run model export
export_models() {
    log_info "Exporting models..."
    
    # Check if model config exists
    if [ ! -f "model_config.yaml" ]; then
        log_warning "model_config.yaml not found, skipping model export"
        return
    fi
    
    # Run model export
    docker run --rm \
        -v $(pwd)/models:/app/models \
        -v $(pwd)/exports:/app/exports \
        ${PROJECT_NAME}:${VERSION} \
        python export_model.py \
        --config model_config.yaml \
        --output-dir /app/exports \
        --formats torchscript onnx
    
    log_success "Models exported"
}

# Run benchmarks
run_benchmarks() {
    log_info "Running benchmarks..."
    
    # Check if exported models exist
    if [ ! -f "exports/model.pt" ] && [ ! -f "exports/model.onnx" ]; then
        log_warning "No exported models found, skipping benchmarks"
        return
    fi
    
    # Run benchmarks for TorchScript
    if [ -f "exports/model.pt" ]; then
        docker run --rm \
            -v $(pwd)/exports:/app/exports \
            -v $(pwd)/benchmarks:/app/benchmarks \
            ${PROJECT_NAME}:${VERSION} \
            python benchmark_inference.py \
            --model-path /app/exports/model.pt \
            --model-type torchscript \
            --output /app/benchmarks/torchscript_results.json
        log_success "TorchScript benchmarks completed"
    fi
    
    # Run benchmarks for ONNX
    if [ -f "exports/model.onnx" ]; then
        docker run --rm \
            -v $(pwd)/exports:/app/exports \
            -v $(pwd)/benchmarks:/app/benchmarks \
            ${PROJECT_NAME}:${VERSION} \
            python benchmark_inference.py \
            --model-path /app/exports/model.onnx \
            --model-type onnx \
            --output /app/benchmarks/onnx_results.json
        log_success "ONNX benchmarks completed"
    fi
}

# Start services
start_services() {
    log_info "Starting services..."
    
    # Stop existing services
    docker-compose down 2>/dev/null || true
    
    # Start services
    docker-compose up -d
    
    log_success "Services started"
    log_info "API available at: http://localhost:8000"
    log_info "Development environment available at: http://localhost:8888"
}

# Run tests
run_tests() {
    log_info "Running tests..."
    
    # Run unit tests
    docker run --rm \
        -v $(pwd):/app \
        ${PROJECT_NAME}:${VERSION} \
        python -m pytest tests/ -v
    
    log_success "Tests completed"
}

# Show status
show_status() {
    log_info "Service status:"
    docker-compose ps
    
    log_info "API health check:"
    curl -f http://localhost:8000/health || log_warning "API health check failed"
}

# Clean up
cleanup() {
    log_info "Cleaning up..."
    docker-compose down
    docker system prune -f
    log_success "Cleanup completed"
}

# Main deployment function
deploy() {
    log_info "Starting deployment of ${PROJECT_NAME}..."
    
    check_docker
    setup_directories
    build_images
    export_models
    run_benchmarks
    start_services
    run_tests
    show_status
    
    log_success "Deployment completed successfully!"
    log_info "Access the application at: http://localhost:8000"
    log_info "View API documentation at: http://localhost:8000/docs"
}

# Parse command line arguments
case "${1:-deploy}" in
    "deploy")
        deploy
        ;;
    "build")
        check_docker
        build_images
        ;;
    "export")
        check_docker
        export_models
        ;;
    "benchmark")
        check_docker
        run_benchmarks
        ;;
    "start")
        check_docker
        start_services
        ;;
    "test")
        check_docker
        run_tests
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
        echo "  deploy     - Full deployment (default)"
        echo "  build      - Build Docker images"
        echo "  export     - Export models"
        echo "  benchmark  - Run benchmarks"
        echo "  start      - Start services"
        echo "  test       - Run tests"
        echo "  status     - Show service status"
        echo "  cleanup    - Clean up resources"
        echo "  help       - Show this help"
        ;;
    *)
        log_error "Unknown command: $1"
        echo "Use '$0 help' for available commands"
        exit 1
        ;;
esac
