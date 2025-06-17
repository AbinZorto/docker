# Use Ubuntu as base image for better compatibility with MinerU
FROM ubuntu:22.04

# Set environment variables
# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3.10-venv \
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
    libmagic1 \
    libmagic-dev \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Install pip and uv (following official MinerU installation guide)
RUN python3 -m pip install --upgrade pip
RUN pip install uv

# Install basic Python dependencies first
COPY requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt

# Install MinerU using the EXACT method from official docs
RUN uv pip install --system -U "mineru[core]"

# For GPU support with sglang acceleration (uncomment if using GPU droplets)
# RUN uv pip install --system -U "mineru[all]"

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/uploads /app/outputs /app/cache /app/logs /app/models

# Set permissions
RUN chmod +x /app/start.sh

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start command
CMD ["./start.sh"]
