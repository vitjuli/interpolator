#!/bin/bash
# View Interpoletor Docker logs
# Usage: ./scripts/docker-logs.sh [backend|frontend] [--follow]

set -e

BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

cd "$PROJECT_ROOT"

SERVICE=""
FOLLOW=""

for arg in "$@"; do
    case $arg in
        backend|frontend)
            SERVICE=$arg
            ;;
        --follow|-f)
            FOLLOW="-f"
            ;;
    esac
done

echo "=================================================="
echo "Interpoletor Docker Logs"
echo "=================================================="
echo ""

if ! docker-compose ps | grep -q "Up"; then
    echo -e "${YELLOW}⚠ No services running${NC}"
    echo "Start with: ${BLUE}./scripts/docker-start.sh${NC}"
    exit 1
fi

if [ -n "$SERVICE" ]; then
    echo -e "${BLUE}Showing logs for: ${SERVICE}${NC}"
    if [ -n "$FOLLOW" ]; then
        echo -e "${YELLOW}Following logs (Ctrl+C to stop)...${NC}"
    fi
    echo ""
    docker-compose logs $FOLLOW $SERVICE
else
    echo -e "${BLUE}Showing all service logs${NC}"
    if [ -n "$FOLLOW" ]; then
        echo -e "${YELLOW}Following logs (Ctrl+C to stop)...${NC}"
    fi
    echo ""
    docker-compose logs $FOLLOW
fi
