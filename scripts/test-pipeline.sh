#!/bin/bash
# Test the complete Interpoletor pipeline
# Usage: ./scripts/test-pipeline.sh

set -e

echo "=================================================="
echo "Interpoletor Pipeline Test"
echo "=================================================="
echo ""

GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

cd "$PROJECT_ROOT"

TESTS_PASSED=0
TESTS_FAILED=0

test_endpoint() {
    local name=$1
    local url=$2
    local expected=$3

    echo -n "Testing ${name}... "
    if curl -f -s "$url" | grep -q "$expected"; then
        echo -e "${GREEN}✓ PASS${NC}"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}✗ FAIL${NC}"
        ((TESTS_FAILED++))
    fi
}

echo -e "${BLUE}Checking services...${NC}"
echo ""

if ! curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo -e "${RED}✗ Backend not running${NC}"
    echo "Start with: ${BLUE}./scripts/docker-start.sh${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Backend running${NC}"

if ! curl -f http://localhost:3000 > /dev/null 2>&1; then
    echo -e "${YELLOW}⚠ Frontend not responding${NC}"
else
    echo -e "${GREEN}✓ Frontend running${NC}"
fi

echo ""
echo -e "${BLUE}Testing API endpoints...${NC}"
echo ""

test_endpoint "Health check" "http://localhost:8000/health" "active"
test_endpoint "API docs" "http://localhost:8000/docs" "Interpoletor"
test_endpoint "OpenAPI schema" "http://localhost:8000/openapi.json" "openapi"

echo ""
echo -e "${BLUE}Running backend tests...${NC}"
echo ""

cd "$PROJECT_ROOT/backend"

if command -v pytest &> /dev/null; then
    if PYTHONPATH=. pytest tests/ -q 2>&1 | tee /tmp/pytest.txt; then
        PASSED=$(grep -o "[0-9]* passed" /tmp/pytest.txt | awk '{print $1}')
        echo -e "${GREEN}✓ ${PASSED} tests passed${NC}"
        ((TESTS_PASSED+=$PASSED))
    fi
    rm -f /tmp/pytest.txt
fi

echo ""
echo "=================================================="
echo "Results: ${GREEN}${TESTS_PASSED} passed${NC}, ${RED}${TESTS_FAILED} failed${NC}"
echo "=================================================="

[ $TESTS_FAILED -eq 0 ] && exit 0 || exit 1
