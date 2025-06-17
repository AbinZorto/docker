# Use NVIDIA CUDA runtime with cuDNN for optimized inference
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Install system dependencies (runtime-focused)
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3.10-venv \
    git \
    wget \
    curl \
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
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create app directory
WORKDIR /app

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Install pip and uv
RUN python3 -m pip install --upgrade pip
RUN pip install uv

# Install basic Python dependencies first
COPY requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt

# Install PyTorch with CUDA support first (specific versions for compatibility)
RUN uv pip install --system torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

# Install MinerU CORE (no [all] extras to avoid sglang/flashinfer)
RUN uv pip install --system -U mineru

# Install ONLY the backends you need for T4 efficiency
RUN uv pip install --system transformers accelerate

# Install GPU detection dependencies (CRITICAL for VRAM detection fix)
RUN uv pip install --system pynvml py3nvml nvidia-ml-py3

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/uploads /app/outputs /app/cache /app/logs /app/models

# Set permissions
RUN chmod +x /app/start.sh

# CRITICAL: Set environment variables for GPU detection
ENV CUDA_VISIBLE_DEVICES=0
ENV TORCH_CUDA_ARCH_LIST="7.5"
ENV FORCE_CUDA=1

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start command
CMD ["./start.sh"]
