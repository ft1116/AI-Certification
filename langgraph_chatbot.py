#!/usr/bin/env python3
"""
LangGraph Chatbot for Mineral Insights - FastAPI Integration
Streaming support with conversation memory and Pinecone + Tavily integration
"""

from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Any, Optional, AsyncGenerator, Generator
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
import openai
from dotenv import load_dotenv
import json
import os
import asyncio
from typing import Union
from mapping_agent import DrillingPermitsMapper

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
    conversation_history: List[Dict[str, str]]  # Added for memory support
    streaming_messages: List[Dict[str, str]]  # Added for streaming support
    needs_mapping: bool  # Added for mapping support
    mapping_data: Optional[Dict]  # Added for mapping data
    mapping_summary: Optional[str]  # Added for mapping summary

class LangGraphChatbot:
    """LangGraph-based chatbot with streaming support and conversation memory"""
    
    def __init__(self):
        # Initialize components
        self.llm = ChatAnthropic(model="claude-sonnet-4-5-20250929", temperature=0.3)
        
        # Initialize Pinecone components (lazy initialization)
        self.pc = None
        self.embedder = None
        self.vectorstore = None
        self.openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.embedding_model = "text-embedding-3-small"
        
        # Build the graph
        self.graph = self._create_complete_mineral_graph()
    
    def _initialize_pinecone(self):
        """Lazy initialization of Pinecone components"""
        if self.vectorstore is None:
            try:
                self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
                self.embedder = OpenAIEmbeddings(model="text-embedding-3-small")
                self.vectorstore = PineconeVectorStore.from_existing_index(index_name="forum-db", embedding=self.embedder)
                print("✅ Pinecone initialized successfully")
            except Exception as e:
                print(f"⚠️ Pinecone initialization failed: {e}")
                raise e
    
    def create_embedding(self, text: str) -> List[float]:
        """Create embedding for text using OpenAI"""
        response = self.openai_client.embeddings.create(
            input=text,
            model=self.embedding_model
        )
        return response.data[0].embedding
    
    def _decompose_query(self, query: str) -> List[str]:
        """Decompose complex queries into 2 focused sub-queries"""
        query_lower = query.lower()
        
        # Check if query is complex enough to warrant decomposition
        complexity_indicators = [
            "compare", "versus", "vs", "and", "both", "multiple", "different",
            "between", "across", "various", "several", "all", "each"
        ]
        
        is_complex = any(indicator in query_lower for indicator in complexity_indicators)
        
        if not is_complex:
            # Simple query - return as-is
            return [query]
        
        # Decompose into 2 sub-queries
        sub_queries = []
        
        # Strategy 1: Split on comparison words
        if any(word in query_lower for word in ["compare", "versus", "vs", "between"]):
            # Extract entities to compare
            if "county" in query_lower:
                # Extract county names
                import re
                counties = re.findall(r'(\w+\s+county)', query_lower)
                if len(counties) >= 2:
                    sub_queries = [
                        f"lease rates and activity in {counties[0]}",
                        f"lease rates and activity in {counties[1]}"
                    ]
        
        # Strategy 2: Split on "and" or "both"
        elif "and" in query_lower or "both" in query_lower:
            # Try to split the query
            if "lease" in query_lower and "mineral" in query_lower:
                sub_queries = [
                    "lease offers and rates",
                    "mineral rights purchase offers"
                ]
            elif "texas" in query_lower and "oklahoma" in query_lower:
                sub_queries = [
                    "oil and gas activity in Texas",
                    "oil and gas activity in Oklahoma"
                ]
        
        # Strategy 3: General decomposition
        if not sub_queries:
            # Split into market analysis and specific data
            if "market" in query_lower or "trends" in query_lower:
                sub_queries = [
                    "current market conditions and trends",
                    "specific lease rates and mineral offers"
                ]
            else:
                # Generic split
                sub_queries = [
                    f"general information about {query}",
                    f"specific data and rates for {query}"
                ]
        
        # Ensure we have exactly 2 sub-queries
        if len(sub_queries) != 2:
            sub_queries = [query, query]  # Fallback to original query
        
        return sub_queries
    
    def pinecone_to_langchain_docs(self, pinecone_docs) -> List[Document]:
        """Convert Pinecone results to LangChain Documents (already in correct format)"""
        # PineconeVectorStore already returns Document objects, so we just need to enhance metadata
        documents = []
        
        for doc in pinecone_docs:
            # Enhance metadata if needed
            metadata = doc.metadata.copy()
            
            # Add any additional metadata processing here if needed
            # The documents from Pinecone should already have proper metadata
            
            enhanced_doc = Document(page_content=doc.page_content, metadata=metadata)
            documents.append(enhanced_doc)
        
        return documents
    
    def _create_complete_mineral_graph(self):
        """Create the complete LangGraph for Mineral Insights with Pinecone"""
        
        # Node 1: Retrieve Documents (Single Semantic Search)
        def retrieve_documents(state: MineralQueryState):
            """Multi-query retrieval - decomposes complex queries into sub-queries"""
            query = state["query"]
            
            print(f"🔍 Multi-query retrieval for: '{query}'")
            
            # Initialize Pinecone if not already done
            self._initialize_pinecone()
            
            # Multi-query decomposition
            sub_queries = self._decompose_query(query)
            print(f"📝 Decomposed into {len(sub_queries)} sub-queries: {sub_queries}")
            
            all_docs = []
            
            # Process each sub-query
            for i, sub_query in enumerate(sub_queries):
                try:
                    print(f"🔍 Sub-query {i+1}: '{sub_query}'")
                    # Get documents for this sub-query (fewer per sub-query to avoid overlap)
                    sub_docs = self.vectorstore.similarity_search(sub_query, k=180)
                    print(f"📊 Found {len(sub_docs)} documents for sub-query {i+1}")
                    all_docs.extend(sub_docs)
                except Exception as e:
                    print(f"⚠️ Sub-query {i+1} error: {e}")
            
            # Deduplicate documents
            seen_content = set()
            unique_docs = []
            for doc in all_docs:
                content_hash = hash(doc.page_content[:100])  # Use first 100 chars as identifier
                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    unique_docs.append(doc)
            
            print(f"📚 Total unique documents retrieved: {len(unique_docs)}")
            
            return {"retrieved_documents": unique_docs}
        
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
                
                # Boost based on data type diversity - prioritize your rich data
                data_type = doc.metadata.get("data_type", "unknown")
                if data_type in ["lease_offer", "mineral_offer"]:
                    relevance_score = 1.4  # Offers are most valuable for pricing questions
                elif data_type in ["texas_permit", "oklahoma_permit", "permit", "drilling_permit"]:
                    relevance_score = 1.3  # Permits are very relevant for activity questions
                elif data_type in ["forum_topic", "forum_post"]:
                    relevance_score = 1.2  # Forum discussions often have good context
                elif data_type in ["mineral_rights", "leasing_info"]:
                    relevance_score = 1.1  # General mineral rights info
                
                # Boost for location matches
                county = (doc.metadata.get("county") or "").lower()
                state_name = (doc.metadata.get("state") or "").lower()
                if (county and county in query_lower) or (state_name and state_name in query_lower):
                    relevance_score += 0.3
                
                # Boost for operator matches
                operator = (doc.metadata.get("operator") or doc.metadata.get("entity_name") or "").lower()
                if operator and any(op in operator for op in ["comstock", "antero", "continental", "mewbourne", "eog", "xto", "exxon", "pioneer", "camino", "marathon"]):
                    relevance_score += 0.2
                
                # Boost for recent data (if date available)
                if "date" in doc.metadata or "permit_date" in doc.metadata or "approval_date" in doc.metadata:
                    relevance_score += 0.1  # Recent data is more valuable
                
                # Boost for specific formations mentioned in query
                formation = (doc.metadata.get("formation_name") or "").lower()
                if formation and any(f in query_lower for f in ["woodford", "mississippian", "hunton", "sycamore", "barnett", "haynesville", "eagle ford", "permian"]):
                    relevance_score += 0.3
                
                # Boost for depth information (indicates detailed permit data)
                if doc.metadata.get("total_depth") or doc.metadata.get("depth"):
                    relevance_score += 0.1
                
                # Boost for query-specific keywords
                content_lower = doc.page_content.lower()
                if any(keyword in content_lower for keyword in ["bonus", "royalty", "lease", "mineral", "acre"]):
                    relevance_score += 0.1
                
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
            
            # Take top 35 documents (optimized for speed while maintaining quality)
            filtered_docs = ranked_docs[:35]
            
            print(f"🏆 Top {len(filtered_docs)} documents selected for context")
            
            return {"ranked_documents": filtered_docs}
        
        # Node 3: Generate Answer (Rich Context) - STREAMING VERSION
        def generate_answer(state: MineralQueryState):
            """Generate answer with rich context from all data types - STREAMING"""
            query = state["query"]
            documents = state["ranked_documents"]
            conversation_history = state.get("conversation_history", [])
            
            print(f"🤖 Generating streaming answer using {len(documents)} documents")
            
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
            
            # Build conversation history for context
            history_context = ""
            if conversation_history:
                history_context = "\n\n=== CONVERSATION HISTORY ===\n"
                for msg in conversation_history[-6:]:  # Last 6 messages
                    role = "User" if msg.get("type") == "human" else "Assistant"
                    content = msg.get("content", "")
                    history_context += f"{role}: {content}\n"
            
            # Determine if this is a market/activity/trend question that should use emoji formatting
            query_lower = query.lower()
            market_activity_keywords = [
                "market", "activity", "trends", "outlook", "forecast", "drilling activity", 
                "leasing activity", "current", "recent", "2025", "how is", "what's the market",
                "tell me about activity", "show me", "compare", "analysis"
            ]
            should_use_emoji_format = any(keyword in query_lower for keyword in market_activity_keywords)
            
            # Base system prompt
            base_prompt = """You are a mineral rights expert. Answer questions about mineral rights, 
            oil and gas, drilling permits, lease offers, and mineral offers. Use the provided context 
            to give accurate, helpful answers. 

            IMPORTANT DISTINCTIONS:
            - LEASE OFFERS: These are offers to lease mineral rights for drilling/exploration. They typically include bonus payments per acre and royalty rates. Lease offers are temporary agreements (usually 3-5 years) that give operators the right to drill.
            - MINERAL OFFERS: These are offers to PURCHASE mineral rights outright. They are permanent sales transactions where the buyer acquires ownership of the minerals.

            When discussing pricing:
            - For lease offers: Provide bonus payment ranges (per acre) and typical royalty rates (usually 12.5-25%)
            - For mineral offers: Provide purchase price ranges (per acre) for buying mineral rights
            - Always distinguish clearly between leasing vs. purchasing
            - When possible, provide both general ranges and rough averages
            - Focus ONLY on the specific question asked - do not discuss unrelated topics, locations, or offer types
            - If asked about mineral offers, do NOT mention lease offers unless specifically asked to compare
            - If asked about lease offers, do NOT mention mineral offers unless specifically asked to compare
            - Provide the best available information without mentioning data limitations or counts
            - Stay focused on the geographic area and topic requested in the question
            - Keep responses detailed but focused (avoid unnecessary elaboration)"""
            
            # Add formatting instructions based on question type
            if should_use_emoji_format:
                formatting_instructions = """

            FORMATTING FOR MARKET/ACTIVITY/TREND QUESTIONS:
            Use emojis and structured sections for market analysis, activity summaries, and trend discussions:
            - Start with 📊 for main title
            - Use 🔥 for active areas/operators
            - Use 🎯 for key insights
            - Use 💡 for forward-looking analysis
            - Structure with clear sections and bullet points
            - Include both current data AND trend direction (increasing/decreasing/stable)
            - Explain the drivers behind trends
            - Give forward-looking insights when data supports it
            - Structure market analysis with: Current Market State, Trend Direction, Key Drivers, Future Outlook, Regional Variations"""
            else:
                formatting_instructions = """

            FORMATTING FOR INFORMATIONAL QUESTIONS:
            Use clean, professional formatting without emojis:
            - Clear headings with ##
            - Bullet points for lists
            - Professional, straightforward language
            - Focus on accuracy and clarity
            - Avoid emojis and casual formatting"""
            
            system_prompt = base_prompt + formatting_instructions
            
            user_prompt = f"""Based on the following information about mineral rights, answer the user's question:
{history_context}
{context}

Question: {query}

Provide a helpful, accurate answer based on the context."""
            
            # Prepare messages for streaming
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # Calculate confidence (same logic as before)
            query_lower = query.lower()
            current_keywords = ["current", "recent", "today", "now", "latest", "trend", "market"]
            needs_current_info = any(keyword in query_lower for keyword in current_keywords)
            
            market_trend_keywords = ["trends", "outlook", "forecast", "increasing", "decreasing", 
                                    "market", "prices", "activity", "drilling", "production", "volatility"]
            is_market_trend_query = any(keyword in query_lower for keyword in market_trend_keywords)
            
            specific_play_keywords = ["play", "formation", "shale", "basin", "waynesville", "appalachian", "ohio", "indiana", "kentucky"]
            is_specific_play_query = any(keyword in query_lower for keyword in specific_play_keywords)
            
            shale_play_keywords = ["woodford", "barnett", "haynesville", "eagle ford", 
                                  "permian", "waynesville", "marcellus", "utica", "bakkens", "bossier"]
            is_shale_play_query = any(keyword in query_lower for keyword in shale_play_keywords)
            
            # For streaming, we'll calculate confidence after we get the response
            # For now, set a preliminary confidence - be more generous with pipeline data
            if needs_current_info or is_market_trend_query or is_specific_play_query:
                confidence = 0.6  # Lower threshold for current info queries
            else:
                confidence = 0.8  # Higher confidence for general queries
            
            return {
                "final_answer": "",  # Will be filled by streaming
                "confidence_score": confidence,
                "sources_used": sources_used,
                "streaming_messages": messages  # Store messages for streaming
            }
        
        # Node 4: Decision Point - Check if web search is needed
        def should_search_web(state: MineralQueryState):
            """Decide whether to perform web search based on confidence"""
            confidence = state["confidence_score"]
            
            # Trigger web search for queries that need current information or have lower confidence
            needs_search = confidence < 0.5  # Much more conservative - rely on pipeline data
            
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
                "landman", "landowner", "mineral owner", "surface rights", "working interest",
                "woodford", "barnett", "haynesville", "eagle ford", "permian", "marcellus", 
                "utica", "bakkens", "bossier", "shale", "play", "basin"
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
        
        # Node 6: Enhanced Answer Generation (with web search results) - STREAMING VERSION
        def generate_enhanced_answer(state: MineralQueryState):
            """Generate enhanced answer using both vector store and web search results - STREAMING"""
            query = state["query"]
            documents = state["ranked_documents"]
            web_results = state.get("web_search_results", "")
            conversation_history = state.get("conversation_history", [])
            
            print(f"🤖 Generating enhanced streaming answer using {len(documents)} documents and web results")
            
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
            
            # Build conversation history for context
            history_context = ""
            if conversation_history:
                history_context = "\n\n=== CONVERSATION HISTORY ===\n"
                for msg in conversation_history[-6:]:  # Last 6 messages
                    role = "User" if msg.get("type") == "human" else "Assistant"
                    content = msg.get("content", "")
                    history_context += f"{role}: {content}\n"
            
            # Enhanced prompt
            system_prompt = """You are a mineral rights expert specializing in oil and gas, drilling permits, lease offers, and mineral offers. 

If the query is about mineral rights, oil and gas, drilling, leasing, or related topics, provide a helpful, accurate answer based on the provided context.

IMPORTANT DISTINCTIONS:
- LEASE OFFERS: These are offers to lease mineral rights for drilling/exploration. They typically include bonus payments per acre and royalty rates. Lease offers are temporary agreements (usually 3-5 years) that give operators the right to drill.
- MINERAL OFFERS: These are offers to PURCHASE mineral rights outright. They are permanent sales transactions where the buyer acquires ownership of the minerals.

When discussing pricing:
- For lease offers: Provide bonus payment ranges (per acre) and typical royalty rates (usually 12.5-25%)
- For mineral offers: Provide purchase price ranges (per acre) for buying mineral rights
- Always distinguish clearly between leasing vs. purchasing
- When possible, provide both general ranges and rough averages
- Focus ONLY on the specific question asked - do not discuss unrelated topics, locations, or offer types
- If asked about mineral offers, do NOT mention lease offers unless specifically asked to compare
- If asked about lease offers, do NOT mention mineral offers unless specifically asked to compare
- Provide the best available information without mentioning data limitations or counts
- Stay focused on the geographic area and topic requested in the question
- Keep responses detailed but focused (avoid unnecessary elaboration)

When providing lease rate information, if ranges are given in the answer, add context with recent lease averages:
- Bonus payments typically range from $100-$3,000+ per acre depending on location and activity level
- Royalty rates commonly range from 18-25% (with 3/16 or 18.75% being a traditional baseline)
- Primary lease terms usually 3-5 years
- Include these general market ranges when discussing specific lease offers to provide broader context

When analyzing market trends:
- Provide both current data AND trend direction (increasing/decreasing/stable)
- Include time comparisons when possible (current vs historical)
- Explain the drivers behind trends
- Give forward-looking insights when data supports it
- Structure market analysis with: Current Market State, Trend Direction, Key Drivers, Future Outlook, Regional Variations

If the query is outside your area of expertise (not related to mineral rights, oil and gas, drilling, leasing, etc.), politely explain that you specialize in mineral rights and suggest they ask about topics like:
- Drilling permits and activity
- Lease offers and terms
- Mineral offers and purchases
- Royalty rates and bonus payments
- Mineral rights ownership
- Oil and gas operations
- Land and mineral transactions
- Oil and gas industry news and market trends

If web search results are provided, incorporate that information to give the most current and comprehensive answer possible. The web search includes both specific mineral rights information and general oil & gas industry news, so use both types of information to provide context."""
            
            user_prompt = f"""Based on the following information, answer the user's question:
{history_context}
{context}

Question: {query}

Provide a helpful, accurate answer. If the question is about mineral rights, oil and gas, or related topics, use the context to give a comprehensive answer. If the question is outside these topics, politely redirect to your area of expertise."""
            
            # Prepare messages for streaming
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # Enhanced confidence calculation (boost if web search was used)
            base_confidence = min(1.0, len(documents) / 10.0)
            if web_results and "failed" not in web_results.lower():
                base_confidence = min(1.0, base_confidence + 0.2)  # Boost confidence with web search
            
            return {
                "final_answer": "",  # Will be filled by streaming
                "confidence_score": base_confidence,
                "sources_used": sources_used,
                "streaming_messages": messages  # Store messages for streaming
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

        def check_for_mapping_query(state: MineralQueryState):
            """Simple pass-through - mapping detection handled in API layer"""
            return {
                "needs_mapping": False,
                "mapping_data": None,
                "mapping_summary": None
            }
        
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
        workflow.add_node("check_mapping", check_for_mapping_query)
        
        # Add edges with conditional logic
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "rank")
        workflow.add_edge("rank", "generate")
        workflow.add_edge("generate", "should_search_web")
        
        # Conditional edge: if confidence < 0.5, do web search, otherwise validate
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
        workflow.add_edge("validate", "check_mapping")
        workflow.add_edge("check_mapping", END)
        
        return workflow.compile()
    
    async def stream_query(self, query: str, conversation_history: List[Dict[str, str]] = None) -> AsyncGenerator[str, None]:
        """Stream a query through the LangGraph with conversation memory"""
        if conversation_history is None:
            conversation_history = []
        
        try:
            # Create initial state
            initial_state = {
                "query": query,
                "conversation_history": conversation_history,
                "needs_web_search": False,
                "web_search_results": "",
                "needs_mapping": False,
                "mapping_data": None,
                "mapping_summary": None
            }
            
            # Run the complete graph
            result = self.graph.invoke(initial_state)
            
            # Get the streaming messages
            messages = result.get("streaming_messages", [])
            
            if not messages:
                yield "I apologize, but I encountered an error processing your request."
                return
            
            # Stream the LLM response
            full_response = ""
            async for chunk in self.llm.astream(messages):
                if hasattr(chunk, 'content') and chunk.content:
                    content = chunk.content
                    full_response += content
                    yield content
                elif isinstance(chunk, str):
                    full_response += chunk
                    yield chunk
                
        except Exception as e:
            print(f"⚠️ Streaming error: {e}")
            yield f"I apologize, but I encountered an error while generating the response: {str(e)}"
    
    def run_query(self, query: str, conversation_history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """Run a query through the complete Mineral Insights LangGraph (non-streaming)"""
        if conversation_history is None:
            conversation_history = []
        
        initial_state = {
            "query": query,
            "conversation_history": conversation_history,
            "needs_web_search": False,
            "web_search_results": "",
            "needs_mapping": False,
            "mapping_data": None,
            "mapping_summary": None
        }
        
        result = self.graph.invoke(initial_state)
        
        # Get the streaming messages and generate the answer
        messages = result.get("streaming_messages", [])
        
        if not messages:
            return {
                "answer": "I apologize, but I encountered an error processing your request.",
                "confidence": 0.0,
                "sources": []
            }
        
        # Generate the answer using the LLM
        try:
            response = self.llm.invoke(messages)
            answer = response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            print(f"⚠️ LLM error: {e}")
            answer = "I apologize, but I encountered an error while generating the response."
        
        return {
            "answer": answer,
            "confidence": result["confidence_score"],
            "sources": result["sources_used"],
            "needs_mapping": result.get("needs_mapping", False),
            "mapping_data": result.get("mapping_data"),
            "mapping_summary": result.get("mapping_summary")
        }

# Global chatbot instance (will be initialized when first used)
chatbot = None

def get_chatbot():
    """Get or create the global chatbot instance"""
    global chatbot
    if chatbot is None:
        chatbot = LangGraphChatbot()
    return chatbot

# Convenience functions for FastAPI integration
async def stream_mineral_query(query: str, conversation_history: List[Dict[str, str]] = None) -> AsyncGenerator[str, None]:
    """Stream a mineral rights query using LangGraph"""
    bot = get_chatbot()
    async for chunk in bot.stream_query(query, conversation_history):
        yield chunk

def run_mineral_query(query: str, conversation_history: List[Dict[str, str]] = None) -> Dict[str, Any]:
    """Run a mineral rights query using LangGraph (non-streaming)"""
    bot = get_chatbot()
    return bot.run_query(query, conversation_history)
