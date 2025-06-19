#!/bin/bash
# Optimized Docker + NVIDIA installation for AWS g4dn instances
# Designed for unattended installation via User Data

set -e  # Exit on any error

echo "=== MinerU AWS Docker + NVIDIA Setup ==="
echo "Starting installation at $(date)"

# Update system first
echo "Updating system packages..."
export DEBIAN_FRONTEND=noninteractive
sudo apt-get update
sudo rm /var/cache/debconf/config.dat
sudo apt purge nvidia-*550-server*
sudo dpkg --configure -a
sudo apt-get upgrade -y

# Install required packages
echo "Installing required packages..."
sudo apt-get install -y \
    ca-certificates \
    curl \
    gnupg \
    lsb-release \
    wget \
    unzip \
    awscli

# Install NVIDIA drivers first (critical for g4dn instances)
echo "Installing NVIDIA drivers..."
sudo apt-get install -y ubuntu-drivers-common
sudo ubuntu-drivers autoinstall
sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia
sudo modprobe nvidia

# Add Docker's official GPG key (using latest Docker method)
echo "Adding Docker GPG key..."
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add Docker repository (using latest Docker method)
echo "Adding Docker repository..."
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Add NVIDIA Container Toolkit GPG key and repository
echo "Adding NVIDIA Container Toolkit repository..."
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Update package lists
echo "Updating package lists..."
sudo apt-get update

# Install Docker Engine
echo "Installing Docker Engine..."
sudo apt-get install -y \
    docker-ce \
    docker-ce-cli \
    containerd.io \
    docker-buildx-plugin \
    docker-compose-plugin
sudo apt install docker-compose
# Install NVIDIA Container Toolkit
echo "Installing NVIDIA Container Toolkit..."
sudo apt-get install -y nvidia-container-toolkit

# Configure NVIDIA Container Toolkit
echo "Configuring NVIDIA Container Toolkit..."
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

#########################


mkdir -p ~/mineru-service/
cd ~/mineru-service/
git clone https://github.com/AbinZorto/docker.git
cd docker/mineru-service


# Build and start services
docker-compose up --build -d

# Check GPU access in container
docker-compose exec mineru-api nvidia-smi

# Monitor logs
docker-compose logs -f mineru-api

# Test service
curl http://localhost:8000/health


cat > .env << EOF
# MinerU API Configuration
MINERU_API_KEY=cbe9f548-54fe-4e9c-ada6-15291466d213

# Redis Configuration
REDIS_URL=redis://redis:6379

# File Processing Limits
MAX_FILE_SIZE=100
CACHE_TTL=7200

# Application Settings
DEBUG=false
DOMAIN=mineru.writemine.com

# 🚀 SGLang Client Configuration (NEW)
SGLANG_URL=http://127.0.0.1:30000
MINERU_BACKEND=vlm-sglang-client

# T4-Optimized GPU Configuration
MINERU_DEVICE=cuda
MODEL_CACHE_DIR=/app/models

# CRITICAL FIX: Explicit VRAM settings for T4
MINERU_VIRTUAL_VRAM_SIZE=22528
CUDA_VISIBLE_DEVICES=0
TORCH_CUDA_ARCH_LIST=7.5
FORCE_CUDA=1

# T4-Specific Performance Optimizations
TORCH_CUDNN_V8_API_ENABLED=1
MINERU_MODEL_SOURCE=huggingface
MINERU_BATCH_SIZE=1
MINERU_MAX_WORKERS=1
MINERU_ENABLE_GPU_OPTIMIZE=true

# Memory Management
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
CUDA_CACHE_DISABLE=0

# Fallback settings
MINERU_FALLBACK_TO_CPU=false

# 🚀 SGLang Server Performance Settings (NEW)
SGLANG_MAX_RUNNING_REQUESTS=128
SGLANG_MAX_TOTAL_TOKENS=3000000
SGLANG_MAX_PREFILL_TOKENS=32768
SGLANG_MEM_FRACTION_STATIC=0.85
SGLANG_CHUNKED_PREFILL_SIZE=4096
SGLANG_ENABLE_TORCH_COMPILE=true

# 🔐 SSL and Security Settings (NEW)
SSL_ENABLED=true
HTTPS_REDIRECT=true
NGINX_MAX_BODY_SIZE=100M
NGINX_TIMEOUT=300s

# 📊 Monitoring and Logging (NEW)
LOG_LEVEL=INFO
ENABLE_METRICS=true
HEALTH_CHECK_INTERVAL=30
PERFORMANCE_LOGGING=true

# 🎯 Processing Optimization (NEW)
DEFAULT_PROCESSING_METHOD=auto
ENABLE_FORMULA_PARSING=true
ENABLE_TABLE_PARSING=true
EXTRACT_IMAGES=true
OUTPUT_FORMAT=both
PROCESSING_TIMEOUT=300

# 🔄 Auto-restart and Recovery (NEW)
AUTO_RESTART_SGLANG=true
SGLANG_HEALTH_CHECK_INTERVAL=60
MAX_RESTART_ATTEMPTS=3
RESTART_DELAY_SECONDS=30
EOF
