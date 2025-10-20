#!/bin/bash

# Ensure we're in the correct directory
cd "/Users/fmt116/Desktop/Mineral Insights/mineral-insights"

# Verify we're in the right place
if [ ! -f "package.json" ]; then
    echo "Error: package.json not found. Current directory: $(pwd)"
    exit 1
fi

echo "Starting React server from: $(pwd)"
echo "Package.json found: $(ls package.json)"

# Start the React development server
PORT=3002 npm start
