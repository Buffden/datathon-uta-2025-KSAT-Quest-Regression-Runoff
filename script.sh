#!/bin/bash

# Exit immediately if any command fails
set -e

echo "🔧 Creating virtual environment..."
python3 -m venv venv

echo "📦 Activating virtual environment..."
source venv/bin/activate

echo "⬇️ Installing dependencies from requirements.txt..."
pip install --upgrade pip
pip install -r requirements.txt

echo "🚀 Running main script..."
python main.py

echo "✅ Done! All steps completed successfully."
