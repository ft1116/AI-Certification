#!/usr/bin/env python3
"""
Quick setup script to help configure Pinecone for the advanced retrievers evaluation.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def check_pinecone_setup():
    """Check if Pinecone is properly configured"""
    print("🔍 Checking Pinecone Configuration...")
    
    # Check for API key
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        print("❌ PINECONE_API_KEY not found in environment variables")
        print("   Please set it in your .env file or environment")
        return False
    else:
        print("✅ PINECONE_API_KEY found")
    
    # Check for environment
    environment = os.getenv("PINECONE_ENVIRONMENT")
    if not environment:
        print("❌ PINECONE_ENVIRONMENT not found in environment variables")
        print("   Please set it in your .env file or environment")
        return False
    else:
        print(f"✅ PINECONE_ENVIRONMENT found: {environment}")
    
    # Check for index name
    index_name = os.getenv("PINECONE_INDEX_NAME", "mineral-insights")
    print(f"📋 Using index name: {index_name}")
    
    return True

def test_pinecone_connection():
    """Test the Pinecone connection"""
    try:
        from langchain_pinecone import PineconeVectorStore
        from langchain_openai import OpenAIEmbeddings
        
        print("\n🧪 Testing Pinecone Connection...")
        
        # Initialize embeddings
        embeddings = OpenAIEmbeddings()
        
        # Try to connect to Pinecone
        vectorstore = PineconeVectorStore(
            index_name=os.getenv("PINECONE_INDEX_NAME", "mineral-insights"),
            embedding=embeddings
        )
        
        print("✅ Successfully connected to Pinecone!")
        
        # Test a simple search
        results = vectorstore.similarity_search("test query", k=1)
        print(f"✅ Test search returned {len(results)} results")
        
        return True
        
    except ImportError:
        print("❌ langchain-pinecone not installed")
        print("   Run: pip install langchain-pinecone")
        return False
    except Exception as e:
        print(f"❌ Error connecting to Pinecone: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Pinecone Setup Checker")
    print("=" * 50)
    
    if check_pinecone_setup():
        test_pinecone_connection()
    else:
        print("\n📝 To set up Pinecone:")
        print("1. Get your API key from https://app.pinecone.io/")
        print("2. Add to your .env file:")
        print("   PINECONE_API_KEY=your_api_key_here")
        print("   PINECONE_ENVIRONMENT=your_environment_here")
        print("   PINECONE_INDEX_NAME=your_index_name_here")
        print("3. Install: pip install langchain-pinecone")


