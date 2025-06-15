#!/bin/bash

# Start script for MinerU service

# Set up logging
mkdir -p /app/logs
exec > >(tee -a /app/logs/startup.log) 2>&1

echo "🚀 Starting MinerU PDF Processing Service..."
echo "📅 $(date)"

# Verify MinerU installation using the official method
echo "🔍 Verifying MinerU installation..."
if command -v mineru >/dev/null 2>&1; then
    echo "✅ MinerU command line tool found"
    # Test mineru version
    mineru --version 2>/dev/null && echo "✅ MinerU version check passed" || echo "⚠️ MinerU version check failed"
else
    echo "❌ MinerU command not found"
    exit 1
fi

# Alternative Python module verification
python3 -c "
try:
    import mineru
    print('✅ MinerU Python module available')
except ImportError:
    try:
        # Try alternative imports that might work
        import magic_pdf
        print('✅ magic_pdf module available')
    except ImportError:
        print('⚠️ MinerU Python modules not directly importable (this might be normal)')
"

# Verify dependencies
echo "🔍 Checking dependencies..."
python3 -c "
import sys
required_modules = ['fastapi', 'uvicorn', 'redis', 'PIL', 'numpy']
missing = []
for module in required_modules:
    try:
        __import__(module)
        print(f'✅ {module}')
    except ImportError:
        missing.append(module)
        print(f'❌ {module}')

# Check optional dependencies
optional_modules = ['cv2', 'torch']
for module in optional_modules:
    try:
        __import__(module)
        print(f'✅ {module} (optional)')
    except ImportError:
        print(f'⚠️ {module} (optional) - not available')

if missing:
    print(f'Missing required modules: {missing}')
    sys.exit(1)
"

# Set up directories
echo "📁 Setting up directories..."
mkdir -p /app/uploads /app/outputs /app/cache /app/logs /app/models
chmod 755 /app/uploads /app/outputs /app/cache /app/logs /app/models

# Test GPU availability (optional)
echo "🖥️ Checking GPU availability..."
python3 -c "
try:
    import torch
    print(f'🔥 PyTorch version: {torch.__version__}')
    if torch.cuda.is_available():
        print(f'✅ CUDA available: {torch.cuda.device_count()} GPU(s)')
        for i in range(torch.cuda.device_count()):
            print(f'   GPU {i}: {torch.cuda.get_device_name(i)}')
    else:
        print('⚠️ CUDA not available, using CPU only')
except ImportError:
    print('⚠️ PyTorch not available')
"

# Wait for Redis if configured
if [ -n "$REDIS_URL" ]; then
    echo "⏳ Waiting for Redis..."
    python3 -c "
import redis
import time
import sys
from urllib.parse import urlparse

url = '$REDIS_URL'
parsed = urlparse(url)
host = parsed.hostname or 'localhost'
port = parsed.port or 6379

max_attempts = 30
for i in range(max_attempts):
    try:
        r = redis.Redis(host=host, port=port, socket_timeout=5)
        r.ping()
        print('✅ Redis connection successful')
        break
    except Exception as e:
        if i == max_attempts - 1:
            print(f'❌ Redis connection failed after {max_attempts} attempts: {e}')
            sys.exit(1)
        print(f'⏳ Redis not ready, attempt {i+1}/{max_attempts}...')
        time.sleep(2)
"
fi

# Set environment variables
export PYTHONPATH="/app:$PYTHONPATH"
export TESSDATA_PREFIX="/usr/share/tesseract-ocr/5/tessdata/"

# Start the application
echo "🚀 Starting FastAPI application..."
echo "🌐 Service will be available at http://0.0.0.0:8000"
echo "📖 API docs at http://0.0.0.0:8000/docs"

# Use uvicorn with proper settings for production
exec uvicorn app:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1 \
    --log-level info \
    --access-log \
    --loop asyncio \
    --http h11
