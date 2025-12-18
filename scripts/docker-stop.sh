#!/bin/bash
# Stop Interpoletor Docker services
# Usage: ./scripts/docker-stop.sh [--clean]

set -e

echo "=================================================="
echo "Stopping Interpoletor Services"
echo "=================================================="
echo ""

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

cd "$PROJECT_ROOT"

CLEAN_VOLUMES=false
if [ "$1" = "--clean" ]; then
    CLEAN_VOLUMES=true
    echo -e "${YELLOW}⚠ Clean mode: Will remove volumes${NC}"
    echo ""
fi

echo -e "${BLUE}Stopping containers...${NC}"

if [ "$CLEAN_VOLUMES" = true ]; then
    docker-compose down -v
    echo -e "${GREEN}✓ Services stopped and volumes removed${NC}"
else
    docker-compose down
    echo -e "${GREEN}✓ Services stopped${NC}"
fi

echo ""
echo "To start again: ${BLUE}./scripts/docker-start.sh${NC}"
echo ""
