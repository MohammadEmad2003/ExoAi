# Multi-stage Dockerfile for Cosmic Analysts ExoAI
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories for models and exports
RUN mkdir -p uploads models_registry exports benchmarks extracted_modules

# Expose port
EXPOSE 8000

# Default command (with worker startup)
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]


# Development stage
FROM base as development

# Install development dependencies
RUN pip install --no-cache-dir \
    jupyter \
    ipykernel \
    notebook \
    black \
    flake8 \
    pytest \
    pytest-cov

# Expose Jupyter port
EXPOSE 8888

# Development command
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]


# Production stage
FROM base as production

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser
USER appuser

# Production command
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
