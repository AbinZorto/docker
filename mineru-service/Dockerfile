# Use SGLang base image with CUDA 12.4 support for VLM acceleration
FROM lmsysorg/sglang:v0.4.7-cu124

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_VISIBLE_DEVICES=0

# Install additional system dependencies for MinerU
RUN apt-get update && apt-get install -y \
    redis-server \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-eng \
    libgl1 \
    libglib2.0-0 \
    libjpeg-dev \
    libpng-dev \
    libfreetype6-dev \
    libffi-dev \
    libglu1-mesa \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip with system packages override (required for Ubuntu 24.04)
RUN python3 -m pip install --upgrade --break-system-packages pip setuptools wheel

# Create working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt /app/requirements.txt

# Install Python dependencies with system packages override
RUN pip3 install --break-system-packages -r requirements.txt

# Copy application files
COPY . /app/

# Create necessary directories
RUN mkdir -p /app/uploads /app/outputs /app/cache /tmp/mineru_processing

# Set permissions
RUN chmod -R 755 /app

# Start SGLang server in background
RUN mineru-sglang-server --host 0.0.0.0 --port 8001 --max-running-requests 128 --max-total-tokens 3000000 --max-prefill-tokens 32768 --mem-fraction-static 0.85 --chunked-prefill-size 4096 --enable-torch-compile > /tmp/mineru_server.log 2>&1 &

# Expose ports
EXPOSE 8000 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Copy and set up start script
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Start the application
CMD ["/app/start.sh"]