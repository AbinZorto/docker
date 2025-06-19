#!/bin/bash
# Fixed SSL Setup Script for MinerU with Directory Creation

set -e  # Exit on error

echo "🔐 Starting SSL setup for mineru.writemine.com"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
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

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   print_error "This script must be run as root (use sudo)"
   exit 1
fi

print_info "Setting up SSL for MinerU PDF Processing Service"

# Step 0: Install Nginx if not installed and create required directories
print_status "Checking and installing Nginx..."
if ! command -v nginx &> /dev/null; then
    apt update
    apt install -y nginx
fi

# Create required Nginx directories
print_status "Creating required Nginx directories..."
mkdir -p /etc/nginx/sites-available
mkdir -p /etc/nginx/sites-enabled
mkdir -p /var/www/html

# Ensure Nginx is running
systemctl enable nginx
systemctl start nginx

# Step 1: Create Nginx site configuration
print_status "Creating Nginx site configuration..."
tee /etc/nginx/sites-available/mineru << 'EOF'
# Initial HTTP configuration - will be updated for HTTPS later
server {
    listen 80;
    server_name mineru.writemine.com;
    
    # Temporary location for Let's Encrypt validation
    location /.well-known/acme-challenge/ {
        root /var/www/html;
    }
    
    # Redirect other traffic to HTTPS (will be enabled after SSL setup)
    location / {
        return 301 https://$server_name$request_uri;
    }
}

# Placeholder for HTTPS configuration (will be updated)
server {
    listen 443 ssl http2;
    server_name mineru.writemine.com;
    
    # Temporary self-signed certificate (will be replaced)
    ssl_certificate /etc/ssl/certs/ssl-cert-snakeoil.pem;
    ssl_certificate_key /etc/ssl/private/ssl-cert-snakeoil.key;
    
    # SSL Configuration
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;
    
    # Security headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Frame-Options DENY always;
    add_header X-Content-Type-Options nosniff always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;
    
    # File upload size limit
    client_max_body_size 100M;
    client_body_timeout 300s;
    
    # Proxy configuration for MinerU API
    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header X-Forwarded-Host $host;
        proxy_set_header X-Forwarded-Port $server_port;
        
        # Timeout settings for long processing tasks
        proxy_read_timeout 300s;
        proxy_connect_timeout 75s;
        proxy_send_timeout 300s;
        
        # Cloudflare specific headers
        proxy_set_header CF-Connecting-IP $http_cf_connecting_ip;
        proxy_set_header CF-Ray $http_cf_ray;
        proxy_set_header CF-Visitor $http_cf_visitor;
        
        # WebSocket support (if needed)
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        
        # Buffer settings for large responses
        proxy_buffering on;
        proxy_buffer_size 128k;
        proxy_buffers 4 256k;
        proxy_busy_buffers_size 256k;
    }
    
    # Health check endpoint (bypass proxy for quick checks)
    location /health {
        proxy_pass http://localhost:8000/health;
        proxy_set_header Host $host;
        access_log off;
    }
    
    # Static files (if any)
    location /static/ {
        alias /app/static/;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
    
    # Robots.txt
    location = /robots.txt {
        add_header Content-Type text/plain;
        return 200 "User-agent: *\nDisallow: /\n";
    }
}
EOF

# Step 2: Remove default Nginx site if it exists
print_status "Removing default Nginx site..."
if [ -f /etc/nginx/sites-enabled/default ]; then
    rm -f /etc/nginx/sites-enabled/default
fi

# Step 3: Enable the site
print_status "Enabling Nginx site..."
ln -sf /etc/nginx/sites-available/mineru /etc/nginx/sites-enabled/

# Step 4: Check if nginx.conf includes sites-enabled
print_status "Checking Nginx main configuration..."
if ! grep -q "include /etc/nginx/sites-enabled" /etc/nginx/nginx.conf; then
    print_warning "Adding sites-enabled include to nginx.conf..."
    # Add include directive to http block
    sed -i '/http {/a\        include /etc/nginx/sites-enabled/*;' /etc/nginx/nginx.conf
fi

# Step 5: Test Nginx configuration
print_status "Testing Nginx configuration..."
if nginx -t; then
    print_status "Nginx configuration is valid"
else
    print_error "Nginx configuration test failed"
    print_info "Showing Nginx configuration test output:"
    nginx -t
    exit 1
fi

# Step 6: Restart Nginx
print_status "Restarting Nginx..."
systemctl restart nginx

# Step 7: Install Certbot and Cloudflare plugin
print_status "Installing Certbot with Cloudflare DNS plugin..."
apt update
apt install -y certbot python3-certbot-nginx python3-certbot-dns-cloudflare

# Step 8: Create Cloudflare credentials directory
print_status "Setting up Cloudflare credentials..."
mkdir -p /etc/letsencrypt

# Step 9: Create Cloudflare credentials file
print_status "Creating Cloudflare API credentials..."
tee /etc/letsencrypt/cloudflare.ini << EOF
# Cloudflare API credentials for DNS validation
dns_cloudflare_email = abinzorto@gmail.com
dns_cloudflare_api_key = 8339127207e732cb0ae5fe695bf1442b25f82
EOF

# Step 10: Secure the credentials file
chmod 600 /etc/letsencrypt/cloudflare.ini
print_status "Secured Cloudflare credentials file"

# Step 11: Get SSL certificate using DNS validation
print_status "Requesting SSL certificate from Let's Encrypt..."
print_info "This may take a few minutes while DNS propagates..."

if certbot certonly \
    --dns-cloudflare \
    --dns-cloudflare-credentials /etc/letsencrypt/cloudflare.ini \
    --dns-cloudflare-propagation-seconds 60 \
    -d mineru.writemine.com \
    --agree-tos \
    --email abinzorto@gmail.com \
    --non-interactive; then
    print_status "SSL certificate obtained successfully!"
else
    print_error "Failed to obtain SSL certificate"
    print_info "Please check your Cloudflare API credentials and DNS settings"
    print_info "You can continue with HTTP for now and retry SSL later"
    
    # Ask if user wants to continue without SSL
    read -p "Continue without SSL? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
    
    # Update nginx config to serve HTTP only for now
    print_status "Configuring HTTP-only setup..."
    tee /etc/nginx/sites-available/mineru << 'EOF'
server {
    listen 80;
    server_name mineru.writemine.com;
    
    client_max_body_size 100M;
    client_body_timeout 300s;
    
    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 300s;
        proxy_connect_timeout 75s;
        proxy_send_timeout 300s;
    }
    
    location /health {
        proxy_pass http://localhost:8000/health;
        proxy_set_header Host $host;
        access_log off;
    }
}
EOF
    
    nginx -t && systemctl restart nginx
    print_warning "Setup completed with HTTP only. Visit http://mineru.writemine.com"
    exit 0
fi

# Step 12: Update Nginx configuration with real SSL certificate
print_status "Updating Nginx configuration with SSL certificate..."
tee /etc/nginx/sites-available/mineru << 'EOF'
# HTTP to HTTPS redirect
server {
    listen 80;
    server_name mineru.writemine.com;
    
    # Let's Encrypt validation
    location /.well-known/acme-challenge/ {
        root /var/www/html;
    }
    
    # Redirect all other traffic to HTTPS
    location / {
        return 301 https://$server_name$request_uri;
    }
}

# HTTPS configuration
server {
    listen 443 ssl http2;
    server_name mineru.writemine.com;
    
    # SSL Certificate (Let's Encrypt)
    ssl_certificate /etc/letsencrypt/live/mineru.writemine.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/mineru.writemine.com/privkey.pem;
    
    # SSL Configuration
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-SHA384;
    ssl_prefer_server_ciphers off;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;
    ssl_stapling on;
    ssl_stapling_verify on;
    ssl_trusted_certificate /etc/letsencrypt/live/mineru.writemine.com/chain.pem;
    
    # Security headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains; preload" always;
    add_header X-Frame-Options DENY always;
    add_header X-Content-Type-Options nosniff always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;
    add_header Content-Security-Policy "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; font-src 'self';" always;
    
    # File upload size limit for PDF processing
    client_max_body_size 100M;
    client_body_timeout 300s;
    client_header_timeout 300s;
    
    # Main proxy configuration for MinerU API
    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header X-Forwarded-Host $host;
        proxy_set_header X-Forwarded-Port $server_port;
        
        # Timeout settings for long PDF processing tasks
        proxy_read_timeout 300s;
        proxy_connect_timeout 75s;
        proxy_send_timeout 300s;
        
        # Cloudflare specific headers
        proxy_set_header CF-Connecting-IP $http_cf_connecting_ip;
        proxy_set_header CF-Ray $http_cf_ray;
        proxy_set_header CF-Visitor $http_cf_visitor;
        proxy_set_header CF-IPCountry $http_cf_ipcountry;
        
        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        
        # Buffer settings for large PDF responses
        proxy_buffering on;
        proxy_buffer_size 128k;
        proxy_buffers 4 256k;
        proxy_busy_buffers_size 256k;
        proxy_max_temp_file_size 0;
    }
    
    # Health check endpoint optimization
    location /health {
        proxy_pass http://localhost:8000/health;
        proxy_set_header Host $host;
        access_log off;
        proxy_cache_bypass 1;
        add_header Cache-Control "no-cache, no-store, must-revalidate";
    }
    
    # API status endpoint
    location /status {
        proxy_pass http://localhost:8000/status;
        proxy_set_header Host $host;
        add_header Cache-Control "no-cache, no-store, must-revalidate";
    }
    
    # Static files (if any)
    location /static/ {
        alias /app/static/;
        expires 1y;
        add_header Cache-Control "public, immutable";
        add_header X-Content-Type-Options nosniff;
    }
    
    # Security.txt
    location /.well-known/security.txt {
        add_header Content-Type text/plain;
        return 200 "Contact: security@writemine.com\nExpires: 2025-12-31T23:59:59Z\n";
    }
    
    # Robots.txt
    location = /robots.txt {
        add_header Content-Type text/plain;
        return 200 "User-agent: *\nDisallow: /\n";
    }
    
    # Favicon (prevent 404 errors)
    location = /favicon.ico {
        access_log off;
        log_not_found off;
        return 204;
    }
}
EOF

# Step 13: Test updated Nginx configuration
print_status "Testing updated Nginx configuration..."
if nginx -t; then
    print_status "Updated Nginx configuration is valid"
else
    print_error "Updated Nginx configuration test failed"
    nginx -t
    exit 1
fi

# Step 14: Restart Nginx with SSL
print_status "Restarting Nginx with SSL configuration..."
systemctl restart nginx

# Step 15: Set up automatic certificate renewal
print_status "Setting up automatic certificate renewal..."
(crontab -l 2>/dev/null; echo "0 12 * * * /usr/bin/certbot renew --quiet --deploy-hook 'systemctl reload nginx'") | crontab -

# Step 16: Test certificate renewal
print_status "Testing certificate renewal process..."
certbot renew --dry-run

# Step 17: Check firewall (allow HTTP and HTTPS)
print_status "Configuring firewall for HTTP/HTTPS..."
if command -v ufw &> /dev/null; then
    ufw allow 80/tcp
    ufw allow 443/tcp
    print_status "UFW firewall rules updated"
fi

# Step 18: Verify SSL setup
print_status "Verifying SSL setup..."
sleep 5

print_info "Testing HTTP redirect..."
if curl -s -I http://mineru.writemine.com | grep -q "301"; then
    print_status "HTTP to HTTPS redirect working"
else
    print_warning "HTTP redirect may not be working"
fi

print_info "Testing HTTPS connection..."
if curl -s -o /dev/null -w "%{http_code}" https://mineru.writemine.com/health 2>/dev/null | grep -q "200\|502\|503"; then
    print_status "HTTPS connection established"
else
    print_warning "HTTPS connection may have issues"
fi

# Step 19: Display final information
echo ""
echo "🎉 SSL Setup Complete!"
echo ""
print_status "Your MinerU service is now available at: https://mineru.writemine.com"
print_info "Certificate will automatically renew before expiration"
print_info "Nginx access logs: /var/log/nginx/access.log"
print_info "Nginx error logs: /var/log/nginx/error.log"
print_info "SSL certificate location: /etc/letsencrypt/live/mineru.writemine.com/"

echo ""
echo "🔍 Quick tests you can run:"
echo "  curl -I https://mineru.writemine.com/health"
echo "  curl https://mineru.writemine.com/status"
echo "  openssl s_client -connect mineru.writemine.com:443 -servername mineru.writemine.com"

echo ""
echo "📝 If your MinerU API isn't running yet, start it with:"
echo "  cd /path/to/your/app && python app.py"

echo ""
print_status "SSL setup completed successfully! 🔐✨"