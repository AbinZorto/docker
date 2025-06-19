#!/bin/bash
# Cloudflare Tunnel Setup for Home PC
# This bypasses the need for port forwarding

echo "🌐 Setting up Cloudflare Tunnel for home PC access"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️ $1${NC}"
}

# Install cloudflared
print_info "Installing Cloudflare Tunnel (cloudflared)..."

# Download and install cloudflared
if ! command -v cloudflared &> /dev/null; then
    # For Ubuntu/Debian
    wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
    sudo dpkg -i cloudflared-linux-amd64.deb
    rm cloudflared-linux-amd64.deb
    print_status "Cloudflared installed"
else
    print_status "Cloudflared already installed"
fi

# Authenticate with Cloudflare
print_info "Starting Cloudflare authentication..."
print_warning "A browser window will open. Please login and authorize the tunnel."
cloudflared tunnel login

# Create the tunnel
TUNNEL_NAME="mineru-home-$(date +%s)"
print_info "Creating tunnel: $TUNNEL_NAME"
cloudflared tunnel create $TUNNEL_NAME

# Get tunnel ID
TUNNEL_ID=$(cloudflared tunnel list | grep $TUNNEL_NAME | awk '{print $1}')
print_info "Tunnel ID: $TUNNEL_ID"

# Create tunnel configuration
print_info "Creating tunnel configuration..."
mkdir -p ~/.cloudflared

cat > ~/.cloudflared/config.yml << EOF
tunnel: $TUNNEL_ID
credentials-file: /home/$(whoami)/.cloudflared/$TUNNEL_ID.json

ingress:
  # Route mineru.writemine.com to local Docker container
  - hostname: mineru.writemine.com
    service: http://localhost:8000
    originRequest:
      httpHostHeader: mineru.writemine.com
      connectTimeout: 30s
      tlsTimeout: 30s
      tcpKeepAlive: 30s
      keepAliveConnections: 1024
      keepAliveTimeout: 90s
  
  # Catch-all rule (required)
  - service: http_status:404
EOF

print_status "Tunnel configuration created"

# Route DNS through the tunnel
print_info "Setting up DNS routing..."
cloudflared tunnel route dns $TUNNEL_NAME mineru.writemine.com

# Create systemd service for the tunnel
print_info "Creating systemd service..."
sudo tee /etc/systemd/system/cloudflared-$TUNNEL_NAME.service << EOF
[Unit]
Description=Cloudflare Tunnel for MinerU
After=network.target

[Service]
Type=simple
User=$(whoami)
ExecStart=/usr/local/bin/cloudflared tunnel --config /home/$(whoami)/.cloudflared/config.yml run
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# Enable and start the service
sudo systemctl daemon-reload
sudo systemctl enable cloudflared-$TUNNEL_NAME
sudo systemctl start cloudflared-$TUNNEL_NAME

print_status "Cloudflare Tunnel service started"

# Test the setup
print_info "Testing tunnel connection..."
sleep 10

if systemctl is-active --quiet cloudflared-$TUNNEL_NAME; then
    print_status "Tunnel is running!"
    
    print_info "Testing connection through tunnel..."
    if curl -s -o /dev/null -w "%{http_code}" https://mineru.writemine.com/health | grep -q "200"; then
        print_status "SUCCESS! Your MinerU API is now accessible at https://mineru.writemine.com"
    else
        print_warning "Tunnel is running but connection test failed. This might be normal during initial setup."
        print_info "Try testing again in a few minutes: curl https://mineru.writemine.com/health"
    fi
else
    print_warning "Tunnel service failed to start. Check logs:"
    echo "sudo journalctl -u cloudflared-$TUNNEL_NAME -f"
fi

echo ""
echo "🎉 Cloudflare Tunnel Setup Complete!"
echo ""
echo "📋 Summary:"
echo "  - Tunnel Name: $TUNNEL_NAME"
echo "  - Tunnel ID: $TUNNEL_ID"
echo "  - Your site: https://mineru.writemine.com"
echo ""
echo "🔧 Useful commands:"
echo "  sudo systemctl status cloudflared-$TUNNEL_NAME   # Check tunnel status"
echo "  sudo systemctl restart cloudflared-$TUNNEL_NAME  # Restart tunnel"
echo "  sudo journalctl -u cloudflared-$TUNNEL_NAME -f   # View tunnel logs"
echo "  cloudflared tunnel list                           # List all tunnels"
echo ""
echo "✨ Your MinerU API is now accessible from anywhere without port forwarding!"