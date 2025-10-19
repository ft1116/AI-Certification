#!/usr/bin/env python3
"""
Debug script to test retrieval with 1 question and identify the document conversion issue
"""

import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_core.documents import Document

# Load environment variables
load_dotenv()

def debug_document_conversion():
    """Debug the document conversion issue"""
    print("🔍 Debugging document conversion issue...")
    
    # Initialize clients
    qdrant_client = QdrantClient(url="http://localhost:6333")
    collection_name = "mineral_insights"
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Test question
    test_question = "What are typical lease terms for oil and gas drilling in Texas?"
    
    print(f"📋 Test Question: {test_question}")
    
    # Create vector store
    vectorstore = Qdrant(
        client=qdrant_client,
        collection_name=collection_name,
        embeddings=embedding_model
    )
    
    # Test vector search
    print("\n🔍 Testing vector search...")
    try:
        docs = vectorstore.similarity_search(test_question, k=5)
        print(f"✅ Vector search returned {len(docs)} documents")
        
        for i, doc in enumerate(docs):
            print(f"\n📄 Document {i+1}:")
            print(f"  Content type: {type(doc.page_content)}")
            print(f"  Content length: {len(doc.page_content) if doc.page_content else 0}")
            print(f"  Content preview: {doc.page_content[:100] if doc.page_content else 'None'}...")
            print(f"  Metadata: {doc.metadata}")
            
            # Check for None content
            if doc.page_content is None:
                print("  ❌ ERROR: Document has None content!")
            elif not doc.page_content.strip():
                print("  ⚠️ WARNING: Document has empty content!")
            else:
                print("  ✅ Document content is valid")
                
    except Exception as e:
        print(f"❌ Vector search failed: {e}")
    
    # Test direct Qdrant search
    print("\n🔍 Testing direct Qdrant search...")
    try:
        # Create embedding for the question
        query_embedding = embedding_model.embed_query(test_question)
        
        # Search Qdrant directly
        search_results = qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=5,
            with_payload=True
        )
        
        print(f"✅ Direct Qdrant search returned {len(search_results)} results")
        
        for i, result in enumerate(search_results):
            print(f"\n📄 Result {i+1}:")
            print(f"  Score: {result.score}")
            print(f"  Payload keys: {list(result.payload.keys())}")
            
            # Check searchable_text
            searchable_text = result.payload.get('searchable_text')
            print(f"  Searchable text type: {type(searchable_text)}")
            print(f"  Searchable text length: {len(searchable_text) if searchable_text else 0}")
            print(f"  Searchable text preview: {searchable_text[:100] if searchable_text else 'None'}...")
            
            if searchable_text is None:
                print("  ❌ ERROR: searchable_text is None!")
            elif not searchable_text.strip():
                print("  ⚠️ WARNING: searchable_text is empty!")
            else:
                print("  ✅ searchable_text is valid")
                
    except Exception as e:
        print(f"❌ Direct Qdrant search failed: {e}")

if __name__ == "__main__":
    debug_document_conversion()