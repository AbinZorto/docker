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

# Pre-download MinerU models to reduce startup time (optional but recommended)
# This downloads the 9B model that SGLang uses
RUN echo "🔽 Pre-downloading MinerU models to avoid startup delays..." && \
    python3 -c "import os; from huggingface_hub import hf_hub_download; import logging; logging.basicConfig(level=logging.INFO); model_repo = 'opendatalab/MinerU2.0-2505-0.9B'; files_to_download = ['config.json', 'generation_config.json', 'model.safetensors.index.json', 'tokenizer.json', 'tokenizer_config.json']; [hf_hub_download(repo_id=model_repo, filename=f, cache_dir='/root/.cache/huggingface') for f in files_to_download]; print('✅ Model pre-download completed')" || echo "Model pre-download failed, will download on startup"

# Copy application files
COPY . /app/

# Create necessary directories
RUN mkdir -p /app/uploads /app/outputs /app/cache /tmp/mineru_processing

# Set permissions
RUN chmod -R 755 /app

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