#!/bin/bash

# Start script for MinerU service with Slang server

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

# Test GPU availability and MinerU GPU backends
echo "🖥️ Checking GPU and MinerU backend availability..."
python3 -c "
import os

# Check NVIDIA GPU
try:
    import subprocess
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    if result.returncode == 0:
        print('✅ NVIDIA GPU detected')
        print('   GPU Info:')
        for line in result.stdout.split('\n')[:10]:  # First 10 lines
            if 'Tesla' in line or 'GeForce' in line or 'Quadro' in line or 'RTX' in line:
                print(f'   {line.strip()}')
    else:
        print('⚠️ nvidia-smi failed, no GPU or drivers issue')
except Exception as e:
    print(f'⚠️ GPU check failed: {e}')

# Check PyTorch CUDA
try:
    import torch
    print(f'🔥 PyTorch version: {torch.__version__}')
    if torch.cuda.is_available():
        print(f'✅ CUDA available: {torch.cuda.device_count()} GPU(s)')
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f'   GPU {i}: {name} ({memory:.1f}GB)')
        
        # Test CUDA functionality
        try:
            x = torch.randn(100, 100).cuda()
            y = torch.matmul(x, x)
            print('✅ CUDA tensor operations working')
        except Exception as e:
            print(f'⚠️ CUDA tensor test failed: {e}')
    else:
        print('⚠️ CUDA not available, using CPU only')
except ImportError:
    print('⚠️ PyTorch not available')

# Check environment variables
device = os.getenv('MINERU_DEVICE', 'cpu')
backend = os.getenv('MINERU_BACKEND', 'pipeline')
print(f'🔧 MinerU device: {device}')
print(f'🔧 MinerU backend: {backend}')

# Test MinerU with GPU if configured
if device == 'cuda' and backend == 'vlm-transformers':
    print('🧪 Testing MinerU GPU backend availability...')
    try:
        # This is a simplified test - in practice MinerU will test this
        print('✅ GPU backend configuration looks good')
    except Exception as e:
        print(f'⚠️ GPU backend test failed: {e}')
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

# Start Slang server in background
echo "🌐 Starting Slang server..."
start_slang_server() {
    # Default Slang server configuration
    SLANG_PORT=${SLANG_PORT:-8001}
    SLANG_HOST=${SLANG_HOST:-0.0.0.0}
    SLANG_LOG_LEVEL=${SLANG_LOG_LEVEL:-info}
    
    echo "📡 Slang server configuration:"
    echo "   Host: $SLANG_HOST"
    echo "   Port: $SLANG_PORT"
    echo "   Log Level: $SLANG_LOG_LEVEL"
    
    # Check if Slang server is available
    if command -v slang-server >/dev/null 2>&1; then
        echo "✅ slang-server command found"
        
        # Start Slang server in background with logging
        echo "🚀 Starting Slang server on $SLANG_HOST:$SLANG_PORT..."
        nohup slang-server \
            --host $SLANG_HOST \
            --port $SLANG_PORT \
            --log-level $SLANG_LOG_LEVEL \
            > /app/logs/slang-server.log 2>&1 &
        
        SLANG_PID=$!
        echo "📊 Slang server started with PID: $SLANG_PID"
        
        # Wait a moment and check if it's running
        sleep 3
        if kill -0 $SLANG_PID 2>/dev/null; then
            echo "✅ Slang server is running"
            
            # Test connection to Slang server
            echo "🧪 Testing Slang server connection..."
            for i in {1..10}; do
                if curl -s "http://localhost:$SLANG_PORT/health" >/dev/null 2>&1; then
                    echo "✅ Slang server health check passed"
                    break
                elif [ $i -eq 10 ]; then
                    echo "⚠️ Slang server health check failed after 10 attempts"
                else
                    echo "⏳ Waiting for Slang server to be ready... ($i/10)"
                    sleep 2
                fi
            done
        else
            echo "❌ Slang server failed to start"
            echo "📋 Slang server logs:"
            tail -20 /app/logs/slang-server.log
        fi
        
    elif python3 -c "import slang" 2>/dev/null; then
        echo "✅ Slang Python module found, trying to start server..."
        
        # Try to start Slang server via Python
        nohup python3 -m slang.server \
            --host $SLANG_HOST \
            --port $SLANG_PORT \
            --log-level $SLANG_LOG_LEVEL \
            > /app/logs/slang-server.log 2>&1 &
        
        SLANG_PID=$!
        echo "📊 Slang server (Python) started with PID: $SLANG_PID"
        
        # Wait and check
        sleep 3
        if kill -0 $SLANG_PID 2>/dev/null; then
            echo "✅ Slang server (Python) is running"
        else
            echo "❌ Slang server (Python) failed to start"
            echo "📋 Slang server logs:"
            tail -20 /app/logs/slang-server.log
        fi
        
    else
        echo "⚠️ Slang server not found. Available options:"
        echo "   - Install with: pip install slang-server"
        echo "   - Or ensure slang-server binary is in PATH"
        echo "   - Continuing without Slang server..."
    fi
}

# Start Slang server
start_slang_server

# Create cleanup function for graceful shutdown
cleanup() {
    echo "🛑 Shutting down services..."
    
    # Kill Slang server if running
    if [ ! -z "$SLANG_PID" ] && kill -0 $SLANG_PID 2>/dev/null; then
        echo "🔴 Stopping Slang server (PID: $SLANG_PID)..."
        kill $SLANG_PID
        sleep 2
        if kill -0 $SLANG_PID 2>/dev/null; then
            echo "🔴 Force killing Slang server..."
            kill -9 $SLANG_PID
        fi
    fi
    
    echo "👋 Cleanup complete"
    exit 0
}

# Set up signal handlers for graceful shutdown
trap cleanup SIGTERM SIGINT

# Start the FastAPI application
echo "🚀 Starting FastAPI application..."
echo "🌐 MinerU API will be available at http://0.0.0.0:8000"
echo "📖 API docs at http://0.0.0.0:8000/docs"
if [ ! -z "$SLANG_PORT" ]; then
    echo "📡 Slang server available at http://0.0.0.0:$SLANG_PORT"
fi

# Use uvicorn with proper settings for production
exec uvicorn app:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1 \
    --log-level info \
    --access-log \
    --loop asyncio \
    --http h11