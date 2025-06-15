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

# Install pip and upgrade
RUN python3 -m pip install --upgrade pip setuptools wheel

# Install basic Python dependencies first
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Install MinerU using the correct method
# First install the base dependencies
RUN pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install MinerU from git source (more reliable than pypi for latest version)
RUN pip3 install --no-cache-dir git+https://github.com/opendatalab/MinerU.git

# Alternative: Install from PyPI if git method fails
# RUN pip3 install --no-cache-dir magic-pdf[full-cpu]

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
