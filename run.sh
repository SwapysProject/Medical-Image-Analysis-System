#!/bin/bash
# Medical Image Analysis System - Quick Start Script

echo "ðŸ”¬ Medical Image Analysis System - Quick Start"
echo "=============================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "âŒ Python 3 is required but not installed."
    echo "Please install Python 3.7 or higher."
    exit 1
fi

# Check if pip is available
if ! command -v pip3 &> /dev/null; then
    echo "âŒ pip3 is required but not installed."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "ðŸ“¦ Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "ðŸ”§ Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "â¬‡ï¸  Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "ðŸ“ Creating directories..."
mkdir -p uploads outputs data

# Run the application
echo "ðŸš€ Starting Medical Image Analysis System..."
echo "Open your browser and navigate to: http://localhost:5000"
echo "Press Ctrl+C to stop the server"
echo ""

python app.py