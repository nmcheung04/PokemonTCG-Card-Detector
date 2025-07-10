#!/bin/bash

# Pokemon Card Manager Deployment Script
# This script sets up the application for production deployment

set -e

echo "🚀 Deploying Pokemon Card Manager..."

# Check if we're in the right directory
if [ ! -f "app.py" ]; then
    echo "❌ Error: app.py not found. Please run this script from the webapp directory."
    exit 1
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "❌ Error: .env file not found. Please run setup.py first."
    exit 1
fi

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Check if card database exists
if [ ! -f "../card_database.pkl" ]; then
    echo "⚠️  Warning: card_database.pkl not found in parent directory"
fi

# Create logs directory
mkdir -p logs

# Set environment variables for production
export FLASK_ENV=production
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Start the application with Gunicorn
echo "🌟 Starting application with Gunicorn..."
echo "📱 Application will be available at http://localhost:5000"
echo "🛑 Press Ctrl+C to stop the server"
echo ""

# Run with Gunicorn
# -w 4: 4 worker processes
# -b 0.0.0.0:5000: bind to all interfaces on port 5000
# --access-logfile: log access requests
# --error-logfile: log errors
# --log-level: set log level
# --timeout: worker timeout
# --max-requests: restart workers after this many requests
# --max-requests-jitter: add randomness to max requests
gunicorn \
    -w 4 \
    -b 0.0.0.0:5000 \
    --access-logfile logs/access.log \
    --error-logfile logs/error.log \
    --log-level info \
    --timeout 120 \
    --max-requests 1000 \
    --max-requests-jitter 100 \
    app:app 