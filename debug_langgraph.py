#!/usr/bin/env python3
"""
Debug script for LangGraph + Pinecone integration
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_langgraph_directly():
    """Test LangGraph directly without FastAPI"""
    print("🧪 Testing LangGraph directly...")
    print("=" * 50)
    
    try:
        from langgraph_chatbot import get_chatbot
        
        print("✅ Successfully imported langgraph_chatbot")
        
        # Get chatbot instance
        chatbot = get_chatbot()
        print("✅ Successfully created chatbot instance")
        
        # Test a simple query
        test_query = "What are typical lease terms in Texas?"
        print(f"🔍 Testing query: {test_query}")
        
        # Test non-streaming
        print("📡 Testing non-streaming query...")
        result = chatbot.run_query(test_query)
        print(f"✅ Non-streaming result: {result}")
        
        # Test streaming
        print("📡 Testing streaming query...")
        import asyncio
        
        async def test_streaming():
            async for chunk in chatbot.stream_query(test_query):
                print(f"📝 Chunk: {chunk}")
                break  # Just get first chunk for testing
        
        asyncio.run(test_streaming())
        print("✅ Streaming test completed")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

def test_pinecone_connection():
    """Test Pinecone connection directly"""
    print("\n🧪 Testing Pinecone connection...")
    print("=" * 50)
    
    try:
        from langchain_pinecone import PineconeVectorStore
        from langchain_openai import OpenAIEmbeddings
        from pinecone import Pinecone
        
        print("✅ Successfully imported Pinecone components")
        
        # Test Pinecone connection
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        print("✅ Successfully connected to Pinecone")
        
        # Test vector store
        embedder = OpenAIEmbeddings(model="text-embedding-3-small")
        vectorstore = PineconeVectorStore.from_existing_index(index_name="forum-db", embedding=embedder)
        print("✅ Successfully created vector store")
        
        # Test search
        results = vectorstore.similarity_search("Texas lease terms", k=5)
        print(f"✅ Found {len(results)} documents")
        
        if results:
            print(f"📄 First result: {results[0].page_content[:200]}...")
        
    except Exception as e:
        print(f"❌ Pinecone error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🚀 Starting LangGraph + Pinecone Debug")
    print("=" * 60)
    
    test_pinecone_connection()
    test_langgraph_directly()
