#!/bin/bash
# Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir -p data/snapshots
mkdir -p data/videos

echo "Backend environment setup complete!"