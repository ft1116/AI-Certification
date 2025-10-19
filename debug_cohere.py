#!/usr/bin/env python3
"""
Debug script to test Cohere Rerank integration
"""

import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

# Load environment variables
load_dotenv()

def test_cohere_integration():
    """Test Cohere Rerank integration step by step"""
    print("🔍 Testing Cohere Integration...")
    
    # Test 1: Check Cohere import
    try:
        from langchain_cohere import CohereRerank
        print("✅ CohereRerank import successful")
    except ImportError as e:
        print(f"❌ CohereRerank import failed: {e}")
        return False
    
    # Test 2: Check API key
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        print("❌ COHERE_API_KEY not found")
        return False
    else:
        print("✅ COHERE_API_KEY found")
    
    # Test 3: Test Pinecone connection
    try:
        embeddings = OpenAIEmbeddings()
        vectorstore = PineconeVectorStore(
            index_name=os.getenv("PINECONE_INDEX_NAME", "forum-db"),
            embedding=embeddings
        )
        print("✅ Pinecone connection successful")
    except Exception as e:
        print(f"❌ Pinecone connection failed: {e}")
        return False
    
    # Test 4: Test base retriever
    try:
        base_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        docs = base_retriever.get_relevant_documents("test query")
        print(f"✅ Base retriever working: {len(docs)} documents retrieved")
    except Exception as e:
        print(f"❌ Base retriever failed: {e}")
        return False
    
    # Test 5: Test Cohere reranker
    try:
        compressor = CohereRerank(
            model="rerank-english-v3.0",
            top_n=3
        )
        print("✅ CohereRerank compressor created")
    except Exception as e:
        print(f"❌ CohereRerank compressor failed: {e}")
        return False
    
    # Test 6: Test full integration
    try:
        from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
        retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever
        )
        print("✅ ContextualCompressionRetriever created")
        
        # Test retrieval
        result_docs = retriever.get_relevant_documents("test query")
        print(f"✅ Full retriever working: {len(result_docs)} documents retrieved")
        
        return True
    except Exception as e:
        print(f"❌ Full integration failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Cohere Integration Debugger")
    print("=" * 50)
    
    if test_cohere_integration():
        print("\n🎉 All tests passed! Cohere integration should work.")
    else:
        print("\n❌ Some tests failed. Check the errors above.")
