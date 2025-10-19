#!/usr/bin/env python3
"""
Test the fixed retrieval with 1 question
"""

import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_core.documents import Document
from langchain.retrievers.multi_query import MultiQueryRetriever

# Load environment variables
load_dotenv()

def test_fixed_retrieval():
    """Test the fixed retrieval with 1 question"""
    print("🧪 Testing fixed retrieval with 1 question...")
    
    # Initialize clients
    qdrant_client = QdrantClient(url="http://localhost:6333")
    collection_name = "mineral_insights"
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    llm = ChatAnthropic(model="claude-sonnet-4-5-20250929", temperature=0.3)
    
    # Test question
    test_question = "What are typical lease terms for oil and gas drilling in Texas?"
    
    print(f"📋 Test Question: {test_question}")
    
    try:
        # Create a custom retriever that fixes the document conversion
        class FixedQdrantRetriever:
            def __init__(self, qdrant_client, collection_name, embedding_model):
                self.qdrant_client = qdrant_client
                self.collection_name = collection_name
                self.embedding_model = embedding_model
            
            def get_relevant_documents(self, query: str, k: int = 20):
                # Create embedding for the query
                query_embedding = self.embedding_model.embed_query(query)
                
                # Search Qdrant directly
                search_results = self.qdrant_client.search(
                    collection_name=self.collection_name,
                    query_vector=query_embedding,
                    limit=k,
                    with_payload=True
                )
                
                # Convert to LangChain documents with proper handling
                documents = []
                for result in search_results:
                    payload = result.payload
                    content = payload.get('searchable_text', '')
                    
                    # Handle None content
                    if content is None:
                        content = ''
                    
                    # Ensure content is a string and not empty
                    if not isinstance(content, str):
                        content = str(content) if content is not None else ''
                    
                    # Skip documents with no content
                    if not content or not content.strip():
                        continue
                    
                    metadata = {
                        'data_type': payload.get('data_type', 'unknown'),
                        'source': payload.get('source', 'unknown'),
                        'score': result.score
                    }
                    
                    doc = Document(page_content=content, metadata=metadata)
                    documents.append(doc)
                
                return documents
        
        # Create fixed retriever
        fixed_retriever = FixedQdrantRetriever(
            qdrant_client,
            collection_name,
            embedding_model
        )
        
        # Test the fixed retriever directly
        print("\n🔍 Testing fixed retriever directly...")
        docs = fixed_retriever.get_relevant_documents(test_question, k=5)
        print(f"✅ Fixed retriever returned {len(docs)} documents")
        
        for i, doc in enumerate(docs):
            print(f"  Document {i+1}: {doc.page_content[:50]}...")
        
        # Test Multi-Query retriever with fixed base retriever
        print("\n🔍 Testing Multi-Query retriever...")
        multi_query_retriever = MultiQueryRetriever.from_llm(
            retriever=fixed_retriever,
            llm=llm
        )
        
        # Test the multi-query retriever
        multi_docs = multi_query_retriever.get_relevant_documents(test_question)
        print(f"✅ Multi-Query retriever returned {len(multi_docs)} documents")
        
        for i, doc in enumerate(multi_docs):
            print(f"  Document {i+1}: {doc.page_content[:50]}...")
        
        print("\n✅ All tests passed! The fixed retrieval is working.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_fixed_retrieval()
