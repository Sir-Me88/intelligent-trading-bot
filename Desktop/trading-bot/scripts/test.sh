#!/bin/bash

# Run tests for Forex Trading Bot

set -e

echo "🧪 Running Forex Trading Bot tests..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Run linting
echo "🔍 Running code linting..."
ruff check src/ tests/ || true

# Run type checking
echo "🔍 Running type checking..."
mypy src/ --ignore-missing-imports || true

# Run tests
echo "🧪 Running unit tests..."
pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html

echo "✅ Tests completed!"
echo ""
echo "📊 Coverage report generated in htmlcov/index.html"