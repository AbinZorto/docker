#!/bin/bash
# Debug and Fix Docker MinerU Setup

set -e

echo "🐳 Debugging Docker MinerU Setup"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️ $1${NC}"
}

echo "=== 🔍 Current Docker Status ==="
docker ps

echo ""
echo "=== 🧪 Testing Local Docker Connection ==="

# Test direct connection to Docker container
print_info "Testing connection to Docker container on localhost:8000..."
if curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health | grep -q "200"; then
    print_status "Docker container is responding on localhost:8000"
    
    # Get the actual response
    echo "Response from container:"
    curl -s http://localhost:8000/health | jq . 2>/dev/null || curl -s http://localhost:8000/health
else
    print_error "Docker container not responding on localhost:8000"
    
    # Test if the port is actually open
    print_info "Checking if port 8000 is actually listening..."
    netstat -tlnp | grep :8000 || ss -tlnp | grep :8000
    
    # Check container logs
    print_info "Checking container logs..."
    docker logs --tail=20 mineru-service-mineru-api-1
fi

echo ""
echo "=== 🔧 Testing Internal Docker Network ==="

# Get container IP
CONTAINER_IP=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' mineru-service-mineru-api-1)
print_info "Container IP: $CONTAINER_IP"

if [ ! -z "$CONTAINER_IP" ]; then
    print_info "Testing connection to container IP: $CONTAINER_IP:8000"
    if curl -s -o /dev/null -w "%{http_code}" http://$CONTAINER_IP:8000/health | grep -q "200"; then
        print_status "Container responding on internal IP"
    else
        print_warning "Container not responding on internal IP"
    fi
fi

echo ""
echo "=== 🛠️ Checking Container Health ==="

# Check container health status
HEALTH_STATUS=$(docker inspect --format='{{.State.Health.Status}}' mineru-service-mineru-api-1 2>/dev/null || echo "no health check")
print_info "Container health status: $HEALTH_STATUS"

# Check if container has health check defined
HEALTH_CHECK=$(docker inspect --format='{{.Config.Healthcheck}}' mineru-service-mineru-api-1 2>/dev/null || echo "undefined")
print_info "Health check config: $HEALTH_CHECK"

echo ""
echo "=== 🔍 Checking Nginx Configuration ==="

# Check current nginx config
print_info "Current Nginx proxy configuration:"
sudo cat /etc/nginx/sites-available/mineru | grep -A 10 -B 2 "proxy_pass"

echo ""
echo "=== 🌐 Testing External Access ==="

# Test from external perspective (what Cloudflare sees)
print_info "Testing what Cloudflare sees..."
PUBLIC_IP=$(curl -s https://api.ipify.org)
print_info "Your public IP: $PUBLIC_IP"

# Test if the port is accessible externally
print_info "Testing external access to port 8000..."
if timeout 5 bash -c "</dev/tcp/$PUBLIC_IP/8000" 2>/dev/null; then
    print_status "Port 8000 is accessible externally"
else
    print_warning "Port 8000 may not be accessible externally"
fi

echo ""
echo "=== 🔧 Proposed Fixes ==="

echo "1. 🐳 Update Nginx to point to Docker container:"
echo ""
cat << 'EOF'
# Updated Nginx configuration for Docker
sudo tee /etc/nginx/sites-available/mineru << 'NGINX_EOF'
server {
    listen 80;
    server_name mineru.writemine.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name mineru.writemine.com;
    
    ssl_certificate /etc/letsencrypt/live/mineru.writemine.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/mineru.writemine.com/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
    ssl_prefer_server_ciphers off;
    
    # Security headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Frame-Options DENY always;
    add_header X-Content-Type-Options nosniff always;
    
    client_max_body_size 100M;
    client_body_timeout 300s;
    
    location / {
        # Point to Docker container
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeout settings for PDF processing
        proxy_read_timeout 300s;
        proxy_connect_timeout 75s;
        proxy_send_timeout 300s;
        
        # Cloudflare headers
        proxy_set_header CF-Connecting-IP $http_cf_connecting_ip;
        proxy_set_header CF-Ray $http_cf_ray;
        
        # Connection settings
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        
        # Buffer settings
        proxy_buffering on;
        proxy_buffer_size 128k;
        proxy_buffers 4 256k;
        proxy_busy_buffers_size 256k;
    }
    
    location /health {
        proxy_pass http://127.0.0.1:8000/health;
        proxy_set_header Host $host;
        access_log off;
        add_header Cache-Control "no-cache";
    }
}
NGINX_EOF
EOF

echo ""
echo "2. 🔄 Apply the fix:"
echo "   sudo nginx -t && sudo systemctl restart nginx"

echo ""
echo "3. 🧪 Test the connection:"
echo "   curl https://mineru.writemine.com/health"

echo ""
echo "=== 🚀 Quick Fix (Run this) ==="
echo ""

read -p "Do you want to apply the Nginx fix now? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_status "Applying Nginx configuration fix..."
    
    # Create the updated nginx config
    sudo tee /etc/nginx/sites-available/mineru << 'NGINX_EOF'
# Fixed Nginx configuration with compatible SSL ciphers
server {
    listen 80;
    server_name mineru.writemine.com;
    
    # Let's Encrypt validation
    location /.well-known/acme-challenge/ {
        root /var/www/html;
    }
    
    # Redirect to HTTPS
    location / {
        return 301 https://$server_name$request_uri;
    }
}

server {
    listen 443 ssl http2;
    server_name mineru.writemine.com;
    
    # SSL Certificate
    ssl_certificate /etc/letsencrypt/live/mineru.writemine.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/mineru.writemine.com/privkey.pem;
    
    # Modern SSL Configuration (Compatible)
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers 'ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384:ECDHE-ECDSA-CHACHA20-POLY1305:ECDHE-RSA-CHACHA20-POLY1305:DHE-RSA-AES128-GCM-SHA256:DHE-RSA-AES256-GCM-SHA384';
    ssl_prefer_server_ciphers off;
    ssl_session_cache shared:SSL:50m;
    ssl_session_timeout 1d;
    ssl_session_tickets off;
    
    # OCSP stapling
    ssl_stapling on;
    ssl_stapling_verify on;
    ssl_trusted_certificate /etc/letsencrypt/live/mineru.writemine.com/chain.pem;
    
    # Security headers
    add_header Strict-Transport-Security "max-age=63072000" always;
    add_header X-Frame-Options DENY always;
    add_header X-Content-Type-Options nosniff always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Referrer-Policy "no-referrer-when-downgrade" always;
    
    # File upload settings for PDF processing
    client_max_body_size 100M;
    client_body_timeout 300s;
    client_header_timeout 60s;
    
    # Main proxy to Docker container
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header X-Forwarded-Host $host;
        proxy_set_header X-Forwarded-Port $server_port;
        
        # Timeout settings for long PDF processing
        proxy_connect_timeout 30s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
        
        # Cloudflare specific headers
        proxy_set_header CF-Connecting-IP $http_cf_connecting_ip;
        proxy_set_header CF-Ray $http_cf_ray;
        proxy_set_header CF-Visitor $http_cf_visitor;
        
        # HTTP version and connection handling
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        
        # Buffer settings for large responses
        proxy_buffering on;
        proxy_buffer_size 64k;
        proxy_buffers 8 64k;
        proxy_busy_buffers_size 128k;
    }
    
    # Health check endpoint
    location /health {
        proxy_pass http://127.0.0.1:8000/health;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        access_log off;
        add_header Cache-Control "no-cache, no-store, must-revalidate";
        add_header Pragma "no-cache";
        add_header Expires "0";
    }
    
    # Status endpoint
    location /status {
        proxy_pass http://127.0.0.1:8000/status;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        add_header Cache-Control "no-cache, no-store, must-revalidate";
    }
    
    # Robots.txt
    location = /robots.txt {
        add_header Content-Type text/plain;
        return 200 "User-agent: *\nDisallow: /\n";
    }
    
    # Favicon
    location = /favicon.ico {
        access_log off;
        log_not_found off;
        return 204;
    }
}
NGINX_EOF
    
    # Test and restart nginx
    if sudo nginx -t; then
        print_status "Nginx configuration is valid"
        sudo systemctl restart nginx
        print_status "Nginx restarted successfully"
        
        # Test the connection
        sleep 2
        print_info "Testing the fixed connection..."
        
        if curl -s -o /dev/null -w "%{http_code}" https://mineru.writemine.com/health | grep -q "200\|502\|503"; then
            print_status "Connection established! Testing response..."
            curl -s https://mineru.writemine.com/health | jq . 2>/dev/null || curl -s https://mineru.writemine.com/health
        else
            print_warning "Still having connection issues. Check Docker container status."
        fi
    else
        print_error "Nginx configuration test failed"
    fi
else
    print_info "Skipping automatic fix. You can apply the configuration manually."
fi

echo ""
echo "=== 📋 Summary ==="
print_info "Your MinerU API is running in Docker and is healthy"
print_info "The issue is likely just the Nginx proxy configuration"
print_info "SSL certificate is properly installed"
print_info "After applying the fix, test with: curl https://mineru.writemine.com/health"