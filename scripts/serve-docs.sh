#!/bin/bash
# Serve Sphinx documentation via HTTP server
# Usage: ./scripts/serve-docs.sh [port]

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

PORT=${1:-8080}

echo "=================================================="
echo "Serving Interpoletor Documentation"
echo "=================================================="
echo ""

if [ ! -d "$PROJECT_ROOT/docs/build/html" ]; then
    echo -e "${YELLOW}⚠ Documentation not built yet${NC}"
    echo ""
    echo "Building documentation..."
    "$SCRIPT_DIR/build-docs.sh"
fi

cd "$PROJECT_ROOT/docs/build/html"

echo -e "${BLUE}Starting HTTP server on port ${PORT}...${NC}"
echo ""

if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo -e "${RED}Error: Port ${PORT} already in use${NC}"
    echo "Try: ${BLUE}./scripts/serve-docs.sh 8081${NC}"
    exit 1
fi

echo "=================================================="
echo -e "📚 URL: ${GREEN}http://localhost:${PORT}${NC}"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop${NC}"
echo "=================================================="
echo ""

if command -v open &> /dev/null; then
    sleep 1 && open "http://localhost:${PORT}" 2>/dev/null &
elif command -v xdg-open &> /dev/null; then
    sleep 1 && xdg-open "http://localhost:${PORT}" 2>/dev/null &
fi

python3 -m http.server $PORT
