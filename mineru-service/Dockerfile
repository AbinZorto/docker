# Use Ubuntu as base image for better compatibility with MinerU
FROM ubuntu:22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    curl \
    build-essential \
    libssl-dev \
    libffi-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libwebp-dev \
    libopencv-dev \
    tesseract-ocr \
    tesseract-ocr-eng \
    poppler-utils \
    libpoppler-cpp-dev \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Install uv for faster package installation (as recommended by MinerU)
RUN pip3 install --upgrade pip uv

# Install Python dependencies
COPY requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt

# Install MinerU with core dependencies (following official installation)
RUN uv pip install --system --no-cache "mineru[core]"

# For GPU support with sglang acceleration (uncomment if using GPU droplets)
# RUN uv pip install --system --no-cache "mineru[all]"

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/uploads /app/outputs /app/cache

# Set permissions
RUN chmod +x /app/start.sh

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start command
CMD ["./start.sh"] 