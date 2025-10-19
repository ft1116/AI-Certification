#!/usr/bin/env python3
"""
Simple test with 1 question using our own document conversion
"""

import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain_anthropic import ChatAnthropic

# Load environment variables
load_dotenv()

def test_simple_retrieval():
    """Test retrieval with our own document conversion"""
    print("🧪 Testing simple retrieval with 1 question...")
    
    # Initialize clients
    qdrant_client = QdrantClient(url="http://localhost:6333")
    collection_name = "mineral_insights"
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    llm = ChatAnthropic(model="claude-sonnet-4-5-20250929", temperature=0.3)
    
    # Test question
    test_question = "What are typical lease terms for oil and gas drilling in Texas?"
    
    print(f"📋 Test Question: {test_question}")
    
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
        
        print(f"✅ Found {len(search_results)} relevant documents")
        
        # Convert to LangChain documents using our method
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
                print(f"⚠️ Skipping document with no content")
                continue
            
            metadata = {
                'data_type': payload.get('data_type', 'unknown'),
                'source': payload.get('source', 'unknown'),
                'score': result.score
            }
            
            # Add rich metadata based on data type
            data = payload.get('data', {})
            if payload.get('data_type') == 'texas_permit':
                metadata.update({
                    'operator': data.get('Operator', ''),
                    'county': data.get('County_Name', ''),
                    'state': data.get('State', ''),
                    'well_number': data.get('Well_Number', ''),
                    'formation': data.get('Formation_Name', ''),
                    'api_number': data.get('API_Number', '')
                })
            elif payload.get('data_type') == 'oklahoma_permit':
                metadata.update({
                    'operator': data.get('Entity_Name', ''),
                    'county': data.get('County', ''),
                    'state': data.get('State', ''),
                    'well_name': data.get('Well_Name', ''),
                    'formation': data.get('Formation_Name', ''),
                    'api_number': data.get('API_Number', '')
                })
            elif payload.get('data_type') in ['mineral_offer', 'lease_offer']:
                metadata.update({
                    'county': data.get('county', ''),
                    'state': data.get('state', ''),
                    'price_per_acre': data.get('price_per_acre', ''),
                    'total_acres': data.get('total_acres', ''),
                    'operator': data.get('operator', ''),
                    'buyer': data.get('buyer', '')
                })
            elif payload.get('data_type') in ['forum_topic', 'forum_post']:
                metadata.update({
                    'title': payload.get('title', ''),
                    'author': payload.get('author', ''),
                    'category': payload.get('category', ''),
                    'url': payload.get('url', ''),
                    'replies': payload.get('replies', 0),
                    'views': payload.get('views', 0)
                })
            
            doc = Document(page_content=content, metadata=metadata)
            documents.append(doc)
            print(f"✅ Created document: {content[:50]}...")
        
        print(f"\n📚 Successfully created {len(documents)} documents")
        
        # Generate answer
        if documents:
            contexts = [doc.page_content for doc in documents]
            context = "\n\n".join([f"Document {i+1}:\n{doc.page_content[:400]}..." for i, doc in enumerate(documents[:3])])
            
            system_prompt = """You are a mineral rights expert. Answer questions about mineral rights, 
            oil and gas, drilling permits, lease offers, and related topics. Use the provided context 
            to give accurate, helpful answers."""
            
            user_prompt = f"""Based on the following information about mineral rights, answer the user's question:

{context}

Question: {test_question}

Provide a helpful, accurate answer based on the context."""
            
            print(f"\n🤖 Generating answer...")
            response = llm.invoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ])
            
            print(f"\n✅ Answer: {response.content}")
            
            # Simple evaluation
            print(f"\n📊 Evaluation:")
            print(f"  - Documents retrieved: {len(documents)}")
            print(f"  - Answer length: {len(response.content)} characters")
            print(f"  - Context coverage: {'Good' if len(context) > 500 else 'Limited'}")
            
        else:
            print("❌ No valid documents found")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_simple_retrieval()

