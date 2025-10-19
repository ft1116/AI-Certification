#!/usr/bin/env python3
"""
Setup and run script for Advanced Retrievers Evaluation

This script handles setup and runs the comprehensive evaluation.
"""

import os
import subprocess
import sys
from dotenv import load_dotenv

def check_requirements():
    """Check if required packages are installed"""
    print("🔍 Checking requirements...")
    
    required_packages = [
        "langchain",
        "langchain_openai", 
        "langchain_anthropic",
        "langchain_community",
        "rank_bm25",
        "pandas",
        "qdrant_client"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️ Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install -r requirements_advanced_retrievers.txt")
        return False
    
    print("✅ All required packages are installed!")
    return True

def check_api_keys():
    """Check if required API keys are set"""
    print("\n🔑 Checking API keys...")
    
    load_dotenv()
    
    required_keys = {
        "OPENAI_API_KEY": "OpenAI (for embeddings)",
        "ANTHROPIC_API_KEY": "Anthropic (for LLM)"
    }
    
    optional_keys = {
        "COHERE_API_KEY": "Cohere (for reranking - optional)"
    }
    
    missing_required = []
    
    for key, description in required_keys.items():
        if os.getenv(key):
            print(f"  ✅ {key} - {description}")
        else:
            print(f"  ❌ {key} - {description}")
            missing_required.append(key)
    
    for key, description in optional_keys.items():
        if os.getenv(key):
            print(f"  ✅ {key} - {description}")
        else:
            print(f"  ⚠️ {key} - {description} (optional)")
    
    if missing_required:
        print(f"\n❌ Missing required API keys: {', '.join(missing_required)}")
        print("Add them to your .env file")
        return False
    
    print("✅ All required API keys are set!")
    return True

def check_qdrant():
    """Check if Qdrant is running"""
    print("\n🗄️ Checking Qdrant connection...")
    
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(url="http://localhost:6333")
        collections = client.get_collections()
        print("  ✅ Qdrant is running and accessible")
        
        # Check if mineral_insights collection exists
        collection_names = [col.name for col in collections.collections]
        if "mineral_insights" in collection_names:
            print("  ✅ mineral_insights collection exists")
            return True
        else:
            print("  ❌ mineral_insights collection not found")
            print("  Run your data ingestion first: python ingest_all_data_to_qdrant.py")
            return False
            
    except Exception as e:
        print(f"  ❌ Qdrant connection failed: {e}")
        print("  Start Qdrant with: docker run -p 6333:6333 qdrant/qdrant")
        return False

def run_evaluation():
    """Run the advanced retrievers evaluation"""
    print("\n🚀 Running Advanced Retrievers Evaluation...")
    print("="*60)
    
    try:
        from advanced_retrievers_evaluation import main
        main()
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        return False
    
    return True

def main():
    """Main setup and run function"""
    print("🔧 Advanced Retrievers Evaluation Setup & Run")
    print("="*60)
    
    # Check requirements
    if not check_requirements():
        print("\n❌ Setup incomplete. Please install missing packages.")
        return
    
    # Check API keys
    if not check_api_keys():
        print("\n❌ Setup incomplete. Please set missing API keys.")
        return
    
    # Check Qdrant
    if not check_qdrant():
        print("\n❌ Setup incomplete. Please start Qdrant and ingest data.")
        return
    
    print("\n✅ All checks passed! Ready to run evaluation.")
    
    # Ask user if they want to proceed
    response = input("\n🚀 Run the evaluation now? (y/N): ").strip().lower()
    
    if response in ['y', 'yes']:
        success = run_evaluation()
        if success:
            print("\n🎉 Evaluation completed successfully!")
            print("\nNext steps:")
            print("1. Review the results to identify the best retriever")
            print("2. Run: python integrate_best_retriever.py")
            print("3. Modify your mineral_insights_langgraph.py with the best retriever")
        else:
            print("\n❌ Evaluation failed. Check the logs for details.")
    else:
        print("\n👋 Evaluation skipped. Run 'python advanced_retrievers_evaluation.py' when ready.")

if __name__ == "__main__":
    main()

