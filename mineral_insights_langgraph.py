#!/usr/bin/env python3
"""
Complete LangGraph for Mineral Insights - Qdrant Integration
Updated to use Tavily web search with Qdrant vector store
"""

from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.documents import Document
from qdrant_client import QdrantClient
import openai
from dotenv import load_dotenv
import json
import os

# Load environment variables
load_dotenv()

# State definition
class MineralQueryState(TypedDict):
    query: str
    retrieved_documents: List[Document]
    ranked_documents: List[Document]
    final_answer: str
    confidence_score: float
    sources_used: List[str]
    conversation_id: str
    needs_web_search: bool
    web_search_results: str

def create_complete_mineral_graph():
    """Create the complete LangGraph for Mineral Insights with Qdrant"""
    
    # Initialize components
    llm = ChatAnthropic(model="claude-sonnet-4-5-20250929", temperature=0.3)
    
    # Initialize Qdrant client
    qdrant_client = QdrantClient(url="http://localhost:6333")
    collection_name = "mineral_insights"
    openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    embedding_model = "text-embedding-3-small"
    
    def create_embedding(text: str) -> List[float]:
        """Create embedding for text using OpenAI"""
        response = openai_client.embeddings.create(
            input=text,
            model=embedding_model
        )
        return response.data[0].embedding
    
    def qdrant_to_langchain_docs(qdrant_results) -> List[Document]:
        """Convert Qdrant results to LangChain Documents"""
        documents = []
        
        # Handle both QueryResponse and list of results
        if hasattr(qdrant_results, 'points'):
            results = qdrant_results.points
        else:
            results = qdrant_results
        
        for result in results:
            payload = result.payload
            content = payload.get('searchable_text', '')
            
            # Create metadata from payload
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
        
        return documents
    
    # Node 1: Retrieve Documents (Single Semantic Search)
    def retrieve_documents(state: MineralQueryState):
        """Single semantic search retrieval - fast and efficient"""
        query = state["query"]
        
        print(f"🔍 Retrieving documents for: '{query}'")
        
        # Create embedding for the query
        query_embedding = create_embedding(query)
        
        # Single semantic search strategy
        try:
            semantic_results = qdrant_client.query_points(
                collection_name=collection_name,
                query=query_embedding,
                limit=200,
                with_payload=True
            )
            semantic_docs = qdrant_to_langchain_docs(qdrant_results=semantic_results)
            print(f"📊 Found {len(semantic_docs)} semantic matches")
        except Exception as e:
            print(f"⚠️ Semantic search error: {e}")
            semantic_docs = []
        
        print(f"📚 Total documents retrieved: {len(semantic_docs)}")
        
        return {"retrieved_documents": semantic_docs}
    
    # Node 2: Rank Documents (Smart Ranking)
    def rank_documents(state: MineralQueryState):
        """Smart ranking that works for all query types"""
        documents = state["retrieved_documents"]
        
        print(f"🎯 Ranking {len(documents)} documents")
        
        # Deduplicate documents - use more content for better deduplication
        unique_docs = []
        seen_content = set()
        for doc in documents:
            content_hash = hash(doc.page_content[:500])  # Increased from 100 to 500 chars
            if content_hash not in seen_content:
                unique_docs.append(doc)
                seen_content.add(content_hash)
        
        print(f"📋 After deduplication: {len(unique_docs)} unique documents")
        
        # Smart relevance scoring
        ranked_docs = []
        query_lower = state["query"].lower()
        
        for doc in unique_docs:
            relevance_score = 1.0
            
            # Boost based on data type diversity
            data_type = doc.metadata.get("data_type", "unknown")
            if data_type in ["forum_topic", "forum_post"]:
                relevance_score = 1.2  # Forum discussions often have good context
            elif data_type in ["lease_offer", "texas_permit", "oklahoma_permit", "mineral_offer"]:
                relevance_score = 1.1  # Structured data is valuable
            
            # Boost for location matches
            county = (doc.metadata.get("county") or "").lower()
            state_name = (doc.metadata.get("state") or "").lower()
            if (county and county in query_lower) or (state_name and state_name in query_lower):
                relevance_score += 0.3
            
            # Boost for operator matches
            operator = (doc.metadata.get("operator") or "").lower()
            if operator and any(op in operator for op in ["comstock", "antero", "continental", "mewbourne", "eog", "xto", "exxon", "pioneer"]):
                relevance_score += 0.2
            
            # Boost for formation matches
            formation = (doc.metadata.get("formation") or "").lower()
            if formation and any(f in formation for f in ["woodford", "barnett", "eagle ford", "haynesville"]):
                relevance_score += 0.2
            
            # Boost for title matches
            title = (doc.metadata.get("title") or "").lower()
            if title and any(word in title for word in query_lower.split()):
                relevance_score += 0.1
            
            # Add original score from vector search
            original_score = doc.metadata.get("score", 0.5)
            relevance_score += original_score * 0.5
            
            doc.metadata["relevance_score"] = relevance_score
            ranked_docs.append(doc)
        
        # Sort by relevance score
        ranked_docs.sort(key=lambda x: x.metadata.get("relevance_score", 1.0), reverse=True)
        
        # Take top 12 documents (good balance of context and focus)
        filtered_docs = ranked_docs[:12]
        
        print(f"🏆 Top {len(filtered_docs)} documents selected for context")
        
        return {"ranked_documents": filtered_docs}
    
    # Node 3: Generate Answer (Rich Context)
    def generate_answer(state: MineralQueryState):
        """Generate answer with rich context from all data types"""
        query = state["query"]
        documents = state["ranked_documents"]
        
        print(f"🤖 Generating answer using {len(documents)} documents")
        
        # Format context from documents
        context_parts = []
        sources_used = []
        
        # Group documents by type for better context
        forum_docs = [d for d in documents if d.metadata.get("data_type") in ["forum_topic", "forum_post"]]
        lease_docs = [d for d in documents if d.metadata.get("data_type") == "lease_offer"]
        permit_docs = [d for d in documents if d.metadata.get("data_type") in ["texas_permit", "oklahoma_permit"]]
        mineral_docs = [d for d in documents if d.metadata.get("data_type") == "mineral_offer"]
        
        # Add context from each data type
        if forum_docs:
            context_parts.append("=== FORUM DISCUSSIONS ===")
            for doc in forum_docs[:3]:  # Top 3 forum discussions
                title = doc.metadata.get("title", "Forum Discussion")
                author = doc.metadata.get("author", "")
                category = doc.metadata.get("category", "")
                
                # Create more descriptive source
                source_desc = f"Forum: {title}"
                if category:
                    source_desc += f" ({category})"
                if author:
                    source_desc += f" by {author}"
                
                context_parts.append(f"{title}\n{doc.page_content[:400]}...")
                sources_used.append(source_desc)
        
        if lease_docs:
            context_parts.append("=== LEASE OFFERS ===")
            for doc in lease_docs[:3]:  # Top 3 lease offers
                operator = doc.metadata.get("operator", "Unknown Operator")
                county = doc.metadata.get("county", "Unknown County")
                state = doc.metadata.get("state", "")
                price = doc.metadata.get("price_per_acre", "")
                acres = doc.metadata.get("total_acres", "")
                
                # Create more descriptive source
                source_desc = f"Lease: {operator} offering {price} per acre in {county}, {state}"
                if acres:
                    source_desc += f" ({acres} acres)"
                
                context_parts.append(f"{operator} - {county}, {state}\n{doc.page_content[:400]}...")
                sources_used.append(source_desc)
        
        if permit_docs:
            context_parts.append("=== DRILLING PERMITS ===")
            for doc in permit_docs[:3]:  # Top 3 permits
                operator = doc.metadata.get("operator", "Unknown Operator")
                county = doc.metadata.get("county", "Unknown County")
                state = doc.metadata.get("state", "")
                formation = doc.metadata.get("formation", "")
                well_name = doc.metadata.get("well_name", doc.metadata.get("well_number", ""))
                api_number = doc.metadata.get("api_number", "")
                
                # Create more descriptive source
                source_desc = f"Permit: {operator} drilling {formation} formation in {county}, {state}"
                if well_name:
                    source_desc += f" (Well: {well_name})"
                if api_number:
                    source_desc += f" (API: {api_number})"
                
                context_parts.append(f"{operator} - {county}\n{doc.page_content[:400]}...")
                sources_used.append(source_desc)
        
        if mineral_docs:
            context_parts.append("=== MINERAL OFFERS ===")
            for doc in mineral_docs[:2]:  # Top 2 mineral offers
                county = doc.metadata.get("county", "Unknown County")
                state = doc.metadata.get("state", "")
                price = doc.metadata.get("price_per_acre", "")
                acres = doc.metadata.get("total_acres", "")
                
                # Create more descriptive source
                source_desc = f"Mineral: {price} per acre in {county}, {state}"
                if acres:
                    source_desc += f" ({acres} acres)"
                
                context_parts.append(f"{county}, {state} - {price}\n{doc.page_content[:400]}...")
                sources_used.append(source_desc)
        
        context = "\n\n".join(context_parts)
        
        # Effective prompt
        system_prompt = """You are a mineral rights expert. Answer questions about mineral rights, 
        oil and gas, drilling permits, lease offers, and related topics. Use the provided context 
        to give accurate, helpful answers. If the context doesn't contain enough information, 
        say so and suggest what additional information might be helpful."""
        
        user_prompt = f"""Based on the following information about mineral rights, answer the user's question:

{context}

Question: {query}

Provide a helpful, accurate answer based on the context."""
        
        # Generate response
        response = llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])
        
        # Simple confidence calculation - just check if query needs current info
        query_lower = query.lower()
        
        # If query asks for current/recent information, lower confidence to trigger web search
        current_keywords = ["current", "recent", "today", "now", "latest", "trend", "market"]
        needs_current_info = any(keyword in query_lower for keyword in current_keywords)
        
        if needs_current_info:
            confidence = 0.7  # Trigger web search for current info
        else:
            confidence = 0.9  # High confidence for historical/factual queries
        
        return {
            "final_answer": response.content,
            "confidence_score": confidence,
            "sources_used": sources_used
        }
    
    # Node 4: Decision Point - Check if web search is needed
    def should_search_web(state: MineralQueryState):
        """Decide whether to perform web search based on confidence"""
        confidence = state["confidence_score"]
        
        # Trigger web search for queries that need current information or have lower confidence
        needs_search = confidence < 0.8  # More aggressive threshold for web search
        
        print(f"🤔 Confidence: {confidence:.2f}, Needs web search: {needs_search}")
        
        return {
            "needs_web_search": needs_search
        }
    
    # Node 5: Tavily Web Search (Mineral Rights + Oil & Gas News)
    def tavily_web_search(state: MineralQueryState):
        """Perform web search using Tavily focused on mineral rights topics and general oil & gas news"""
        query = state["query"]
        
        print(f"🌐 Performing web search for: '{query}'")
        
        # Check if query is mineral rights related
        mineral_keywords = [
            "mineral rights", "oil", "gas", "drilling", "lease", "royalty", "bonus", 
            "permit", "well", "production", "operator", "acre", "formation", "reservoir",
            "fracking", "horizontal drilling", "vertical drilling", "completion",
            "landman", "landowner", "mineral owner", "surface rights", "working interest"
        ]
        
        query_lower = query.lower()
        is_mineral_related = any(keyword in query_lower for keyword in mineral_keywords)
        
        if not is_mineral_related:
            print("⚠️ Query not related to mineral rights, skipping web search")
            return {
                "web_search_results": "OFF_TOPIC"
            }
        
        try:
            from tavily import TavilyClient
            
            # Initialize Tavily client
            tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
            
            # Perform two searches: specific mineral rights + general oil & gas news
            search_queries = [
                f"{query} mineral rights oil gas drilling lease royalty",
                f"{query} oil gas industry news market prices production"
            ]
            
            all_web_content = []
            
            for search_query in search_queries:
                search_results = tavily.search(
                    query=search_query,
                    search_depth="advanced",
                    max_results=3
                )
                
                # Format search results
                for result in search_results.get('results', []):
                    title = result.get('title', '')
                    content = result.get('content', '')
                    url = result.get('url', '')
                    
                    # Filter results to ensure they're relevant
                    if any(keyword in (title + content).lower() for keyword in mineral_keywords):
                        all_web_content.append(f"Title: {title}\nURL: {url}\nContent: {content}")
            
            # Remove duplicates based on title
            seen_titles = set()
            unique_content = []
            for content in all_web_content:
                title = content.split('\n')[0].replace('Title: ', '')
                if title not in seen_titles:
                    seen_titles.add(title)
                    unique_content.append(content)
            
            web_search_results = "\n\n---\n\n".join(unique_content[:5])  # Limit to 5 results
            
            print(f"🌐 Found {len(unique_content)} relevant web results")
            
            return {
                "web_search_results": web_search_results if unique_content else "NO_RELEVANT_RESULTS"
            }
            
        except Exception as e:
            print(f"⚠️ Tavily search error: {e}")
            return {
                "web_search_results": "SEARCH_FAILED"
            }
    
    # Node 6: Enhanced Answer Generation (with web search results)
    def generate_enhanced_answer(state: MineralQueryState):
        """Generate enhanced answer using both vector store and web search results"""
        query = state["query"]
        documents = state["ranked_documents"]
        web_results = state.get("web_search_results", "")
        
        print(f"🤖 Generating enhanced answer using {len(documents)} documents and web results")
        
        # Format context from documents (same as before)
        context_parts = []
        sources_used = []
        
        # Group documents by type for better context
        forum_docs = [d for d in documents if d.metadata.get("data_type") in ["forum_topic", "forum_post"]]
        lease_docs = [d for d in documents if d.metadata.get("data_type") == "lease_offer"]
        permit_docs = [d for d in documents if d.metadata.get("data_type") in ["texas_permit", "oklahoma_permit"]]
        mineral_docs = [d for d in documents if d.metadata.get("data_type") == "mineral_offer"]
        
        # Add context from each data type
        if forum_docs:
            context_parts.append("=== FORUM DISCUSSIONS ===")
            for doc in forum_docs[:3]:
                title = doc.metadata.get("title", "Forum Discussion")
                context_parts.append(f"{title}\n{doc.page_content[:400]}...")
                sources_used.append(f"Forum: {title}")
        
        if lease_docs:
            context_parts.append("=== LEASE OFFERS ===")
            for doc in lease_docs[:3]:
                title = doc.metadata.get("title", "Lease Offer")
                context_parts.append(f"{title}\n{doc.page_content[:400]}...")
                sources_used.append(f"Lease: {title}")
        
        if permit_docs:
            context_parts.append("=== DRILLING PERMITS ===")
            for doc in permit_docs[:3]:
                operator = doc.metadata.get("operator", "Unknown Operator")
                county = doc.metadata.get("county", "Unknown County")
                context_parts.append(f"{operator} - {county}\n{doc.page_content[:400]}...")
                sources_used.append(f"Permit: {operator} - {county}")
        
        if mineral_docs:
            context_parts.append("=== MINERAL OFFERS ===")
            for doc in mineral_docs[:2]:
                title = doc.metadata.get("title", "Mineral Offer")
                context_parts.append(f"{title}\n{doc.page_content[:400]}...")
                sources_used.append(f"Mineral: {title}")
        
        context = "\n\n".join(context_parts)
        
        # Add web search results if available and relevant
        if web_results and web_results not in ["OFF_TOPIC", "NO_RELEVANT_RESULTS", "SEARCH_FAILED"]:
            context += "\n\n=== RECENT WEB INFORMATION ===\n" + web_results
            sources_used.append("Web Search: Recent mineral rights and oil & gas industry information")
        elif web_results == "OFF_TOPIC":
            # Handle off-topic queries gracefully
            context += "\n\n=== QUERY ASSESSMENT ===\nThis query appears to be outside the scope of mineral rights, oil and gas, drilling permits, and related topics."
        
        # Enhanced prompt
        system_prompt = """You are a mineral rights expert specializing in oil and gas, drilling permits, lease offers, and related topics. 

If the query is about mineral rights, oil and gas, drilling, leasing, or related topics, provide a helpful, accurate answer based on the provided context.

If the query is outside your area of expertise (not related to mineral rights, oil and gas, drilling, leasing, etc.), politely explain that you specialize in mineral rights and suggest they ask about topics like:
- Drilling permits and activity
- Lease offers and terms
- Royalty rates and bonus payments
- Mineral rights ownership
- Oil and gas operations
- Land and mineral transactions
- Oil and gas industry news and market trends

If web search results are provided, incorporate that information to give the most current and comprehensive answer possible. The web search includes both specific mineral rights information and general oil & gas industry news, so use both types of information to provide context."""
        
        user_prompt = f"""Based on the following information, answer the user's question:

{context}

Question: {query}

Provide a helpful, accurate answer. If the question is about mineral rights, oil and gas, or related topics, use the context to give a comprehensive answer. If the question is outside these topics, politely redirect to your area of expertise."""
        
        # Generate response
        response = llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])
        
        # Enhanced confidence calculation (boost if web search was used)
        base_confidence = min(1.0, len(documents) / 10.0)
        if web_results and "failed" not in web_results.lower():
            base_confidence = min(1.0, base_confidence + 0.2)  # Boost confidence with web search
        
        return {
            "final_answer": response.content,
            "confidence_score": base_confidence,
            "sources_used": sources_used
        }
    
    # Node 7: Validate Answer (Simple)
    def validate_answer(state: MineralQueryState):
        """Simple validation"""
        answer = state["final_answer"]
        confidence = state["confidence_score"]
        
        # Only add note if confidence is very low
        if confidence < 0.3:
            answer += "\n\nNote: I found limited information on this topic. You might want to provide more specific details or consult additional sources."
        
        return {"final_answer": answer}
    
    # Build the graph
    workflow = StateGraph(MineralQueryState)
    
    # Add nodes
    workflow.add_node("retrieve", retrieve_documents)
    workflow.add_node("rank", rank_documents)
    workflow.add_node("generate", generate_answer)
    workflow.add_node("should_search_web", should_search_web)
    workflow.add_node("decide_route", lambda state: state)  # Decision node
    workflow.add_node("tavily_search", tavily_web_search)
    workflow.add_node("generate_enhanced", generate_enhanced_answer)
    workflow.add_node("validate", validate_answer)
    
    # Add edges with conditional logic
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "rank")
    workflow.add_edge("rank", "generate")
    workflow.add_edge("generate", "should_search_web")
    
    # Conditional edge: if confidence < 0.6, do web search, otherwise validate
    def route_after_decision(state):
        if state["needs_web_search"]:
            return "tavily_search"
        else:
            return "validate"
    
    workflow.add_conditional_edges(
        "should_search_web",
        route_after_decision,
        {
            "tavily_search": "tavily_search",
            "validate": "validate"
        }
    )
    
    workflow.add_edge("tavily_search", "generate_enhanced")
    workflow.add_edge("generate_enhanced", "validate")
    workflow.add_edge("validate", END)
    
    return workflow.compile()

# Usage functions
def run_mineral_query(query: str, conversation_id: str = "default"):
    """Run a query through the complete Mineral Insights LangGraph"""
    
    graph = create_complete_mineral_graph()
    
    initial_state = {
        "query": query,
        "conversation_id": conversation_id,
        "needs_web_search": False,
        "web_search_results": ""
    }
    
    result = graph.invoke(initial_state)
    
    return {
        "answer": result["final_answer"],
        "confidence": result["confidence_score"],
        "sources": result["sources_used"]
    }

def test_complete_graph():
    """Test the complete graph with sample queries"""
    print("🧪 Testing Complete LangGraph with Qdrant and Tavily...")
    print("=" * 60)
    
    test_queries = [
        "What are typical lease terms in Oklahoma?",
        "Show me Pioneer Natural Resources drilling activity in Texas",
        "What's the market like for mineral rights in Grady County?",
        "How do drilling permits work in Texas?",
        "Compare lease offers between Texas and Oklahoma",
        "What's the weather like today?"  # Off-topic query to test web search routing
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 TEST {i}: {query}")
        print("-" * 40)
        
        try:
            result = run_mineral_query(query)
            print(f"✅ Answer generated with confidence: {result['confidence']:.2f}")
            print(f"📚 Sources used: {len(result['sources'])}")
            print(f"💬 Answer preview: {result['answer'][:200]}...")
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print("\n" + "="*60)

if __name__ == "__main__":
    test_complete_graph()