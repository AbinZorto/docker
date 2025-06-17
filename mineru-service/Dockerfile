# Use NVIDIA CUDA base image with matching versions
FROM nvidia/cuda:12.1-devel-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgcc-s1 \
    libjpeg-dev \
    libpng-dev \
    redis-server \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Install PyTorch with CUDA 12.1 support (matching base image)
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install MinerU with all backends
RUN pip3 install mineru[full]==2.0.5

# Install missing pipeline dependencies
RUN pip3 install \
    fast-langdetect \
    doclayout-yolo \
    ultralytics \
    yolo-nas \
    super-gradients

# Install API dependencies
RUN pip3 install \
    fastapi \
    uvicorn \
    python-multipart \
    redis \
    python-magic \
    pillow \
    numpy

# Copy application files
COPY . /app/

# Expose port
EXPOSE 8000

# Start script
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

CMD ["/app/start.sh"]
