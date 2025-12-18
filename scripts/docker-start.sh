#!/bin/bash
# Start Interpoletor services with Docker Compose
# Usage: ./scripts/docker-start.sh

set -e

echo "=================================================="
echo "Starting Interpoletor Services"
echo "=================================================="
echo ""

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

cd "$PROJECT_ROOT"

if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed${NC}"
    exit 1
fi

if ! docker info &> /dev/null; then
    echo -e "${RED}Error: Docker daemon is not running${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Docker is running${NC}"
echo ""

if [ ! -f "docker-compose.yml" ]; then
    echo -e "${RED}Error: docker-compose.yml not found${NC}"
    exit 1
fi

echo -e "${BLUE}Step 1: Stopping any existing containers...${NC}"
docker-compose down 2>/dev/null || true
echo -e "${GREEN}✓ Cleaned up${NC}"
echo ""

echo -e "${BLUE}Step 2: Building images...${NC}"
docker-compose build

echo ""
echo -e "${BLUE}Step 3: Starting services...${NC}"
docker-compose up -d

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ Services started${NC}"
    echo ""
    echo "=================================================="
    echo "Services Running:"
    echo "=================================================="
    echo ""
    echo -e "🌐 Frontend:  ${GREEN}http://localhost:3000${NC}"
    echo -e "🔧 Backend:   ${GREEN}http://localhost:8000${NC}"
    echo -e "📚 API Docs:  ${GREEN}http://localhost:8000/docs${NC}"
    echo ""
    echo "=================================================="
    echo ""
    echo "Useful commands:"
    echo -e "  ${BLUE}./scripts/docker-logs.sh${NC}     - View logs"
    echo -e "  ${BLUE}./scripts/docker-stop.sh${NC}     - Stop services"
    echo ""
    
    docker-compose ps
else
    echo -e "${RED}✗ Failed to start services${NC}"
    docker-compose logs
    exit 1
fi
