#!/bin/bash

# Navigate to your project directory
cd "$(dirname "$0")"

# Create a virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
  python -m venv venv
fi

# Activate virtual environment
source venv/Scripts/activate

# Install necessary packages
pip install fastapi uvicorn python-multipart torch torchvision jinja2

# Run the FastAPI server
uvicorn main:app --reload
