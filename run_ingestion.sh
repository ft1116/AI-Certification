#!/bin/bash

# Mineral Insights Data Ingestion Script
# This script sets up the environment and runs the data ingestion

set -e  # Exit on any error

echo "=========================================="
echo "Mineral Insights Data Ingestion"
echo "=========================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not installed."
    exit 1
fi

# Check if pip is available
if ! command -v pip3 &> /dev/null; then
    echo "Error: pip3 is required but not installed."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv_ingestion" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv_ingestion
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv_ingestion/bin/activate

# Install requirements
echo "Installing requirements..."
pip install -r requirements_ingestion.txt

# Check OpenAI API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Error: OPENAI_API_KEY environment variable not set"
    echo "Please set your OpenAI API key:"
    echo "  export OPENAI_API_KEY='your-api-key-here'"
    echo ""
    echo "You can get an API key from: https://platform.openai.com/api-keys"
    exit 1
fi

# Check if Qdrant is running
echo "Checking Qdrant connection..."
if ! curl -s http://localhost:6333/collections > /dev/null; then
    echo "Warning: Qdrant is not running on localhost:6333"
    echo "Please start Qdrant first:"
    echo "  docker run -p 6333:6333 qdrant/qdrant"
    echo ""
    echo "Or install and run locally:"
    echo "  pip install qdrant-client[fastembed]"
    echo "  python -m qdrant.http"
    echo ""
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Run tests first
echo "Running ingestion tests..."
python test_ingestion.py

if [ $? -eq 0 ]; then
    echo ""
    echo "Tests passed! Running full ingestion..."
    echo "This may take several minutes depending on data size."
    echo ""
    
    # Run the main ingestion
    python ingest_all_data_to_qdrant.py
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "=========================================="
        echo "✓ Ingestion completed successfully!"
        echo "=========================================="
        echo ""
        echo "Your data is now available in Qdrant for LangGraph usage."
        echo "Collection name: mineral_insights"
        echo "Embedding model: OpenAI text-embedding-3-small (1536 dimensions)"
        echo ""
        echo "To query the data, you can use:"
        echo "  - Qdrant web UI: http://localhost:6333/dashboard"
        echo "  - Python client: from qdrant_client import QdrantClient"
        echo ""
        echo "Note: This ingestion used OpenAI embeddings (estimated cost: ~$0.47)."
        echo "Check the ingestion_stats.json file for exact cost details."
        echo ""
    else
        echo ""
        echo "=========================================="
        echo "✗ Ingestion failed!"
        echo "=========================================="
        echo "Check the logs above for error details."
        exit 1
    fi
else
    echo ""
    echo "=========================================="
    echo "✗ Tests failed!"
    echo "=========================================="
    echo "Please fix the issues above before running ingestion."
    exit 1
fi

# Deactivate virtual environment
deactivate

echo "Done!"
