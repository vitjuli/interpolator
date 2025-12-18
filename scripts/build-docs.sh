#!/bin/bash
# Build Sphinx documentation
# Usage: ./scripts/build-docs.sh

set -e

echo "=================================================="
echo "Building Interpoletor Documentation with Sphinx"
echo "=================================================="
echo ""

GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

echo -e "${BLUE}Project root: ${PROJECT_ROOT}${NC}"
echo ""

cd "$PROJECT_ROOT"

if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 is not installed${NC}"
    exit 1
fi

echo -e "${BLUE}Step 1: Installing Sphinx and dependencies...${NC}"
pip3 install -q sphinx sphinx-rtd-theme 2>/dev/null || true
echo -e "${GREEN}✓ Dependencies installed${NC}"
echo ""

if [ ! -d "docs/source" ]; then
    echo -e "${RED}Error: docs/source directory not found${NC}"
    exit 1
fi

echo -e "${BLUE}Step 2: Cleaning previous builds...${NC}"
rm -rf docs/build
mkdir -p docs/build
echo -e "${GREEN}✓ Cleaned previous builds${NC}"
echo ""

echo -e "${BLUE}Step 3: Building HTML documentation...${NC}"
cd docs
sphinx-build -b html source build/html -W --keep-going

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ Documentation built successfully!${NC}"
    echo ""
    echo "=================================================="
    echo -e "HTML docs: ${GREEN}file://${PROJECT_ROOT}/docs/build/html/index.html${NC}"
    echo ""
    echo "To view: open docs/build/html/index.html"
    echo "Or serve: ./scripts/serve-docs.sh"
    echo ""
    echo "=================================================="
else
    echo ""
    echo -e "${RED}✗ Documentation build failed${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}Done!${NC}"
