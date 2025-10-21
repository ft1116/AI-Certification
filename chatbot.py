"""
Mineral Insights - Proprietary AI Chatbot
Copyright (c) 2024 Frank T. All rights reserved.

This software is proprietary and confidential. 
Unauthorized copying, distribution, or use is strictly prohibited.
For licensing inquiries, contact: frank@mineralinsights.com
"""

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferWindowMemory
import cohere
# Import LangGraph chatbot
from langgraph_chatbot import stream_mineral_query, run_mineral_query

# Simple memory implementation to avoid pydantic compatibility issues
class SimpleChatMessageHistory:
    def __init__(self):
        self.messages = []
    
    def add_user_message(self, message: str):
        self.messages.append({"type": "human", "content": message})
    
    def add_ai_message(self, message: str):
        self.messages.append({"type": "ai", "content": message})
    
    def clear(self):
        self.messages = []
from pinecone import Pinecone
import re
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel
import os
import pickle
from pathlib import Path
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
# Geospatial libraries
try:
    import geopandas as gpd
    from shapely.geometry import Point
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False
    print("Warning: GeoPandas/Shapely not available. Some geospatial features may be limited.")
import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import time
import aiofiles
import json
import asyncio
from typing import Dict, Any, List
from langchain_core.messages import HumanMessage, AIMessage
from PyPDF2 import PdfReader
import tempfile
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import sqlite3
from pathlib import Path
from mapping_agent import DrillingPermitsMapper

# NLP for location detection
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
    print("✅ spaCy NER loaded successfully for location detection")
except Exception as e:
    SPACY_AVAILABLE = False
    print(f"⚠️ spaCy not available: {e}")

# Load environment variables
load_dotenv()

# Initialize geocoder
geolocator = Nominatim(user_agent="mineral_insights_app")

# Geocoding cache
class GeocodingCache:
    def __init__(self, cache_file="geocoding_cache.pkl"):
        self.cache_file = Path(cache_file)
        self.cache = self.load_cache()
    
    def load_cache(self):
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'rb') as f:
                    return pickle.load(f)
            except:
                return {}
        return {}
    
    def save_cache(self):
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
        except:
            pass
    
    def get(self, key):
        return self.cache.get(key)
    
    def set(self, key, value):
        self.cache[key] = value
        self.save_cache()

geocoding_cache = GeocodingCache()

def geocode_location(location_text, max_retries=2):
    """
    Geocode a location string to get coordinates using Nominatim (OpenStreetMap)
    with caching and improved timeout handling
    """
    if not location_text or not location_text.strip():
        return None
    
    location_text = location_text.strip()
    
    # Check cache first
    cache_key = location_text.lower()
    cached_result = geocoding_cache.get(cache_key)
    if cached_result:
        print(f"🎯 Using cached geocoding result for: {location_text}")
        return cached_result
    
    # First, check for well-known US oil and gas regions
    us_oil_gas_regions = {
        'permian basin': [-101.0, 31.5],  # West Texas/New Mexico
        'eagle ford shale': [-98.0, 28.0],  # South Texas
        'barnett shale': [-97.0, 32.0],  # North Texas
        'haynesville shale': [-93.0, 32.0],  # Louisiana/East Texas
        'fayetteville shale': [-92.0, 35.0],  # Arkansas
        'marcellus shale': [-78.0, 40.0],  # Pennsylvania/West Virginia
        'bakken formation': [-102.0, 47.0],  # North Dakota
        'niobrara formation': [-104.0, 40.0],  # Colorado/Wyoming
        'anadarko basin': [-99.0, 35.0],  # Oklahoma/Texas
        'arkoma basin': [-95.0, 35.0],  # Oklahoma/Arkansas
        'ardmore basin': [-97.0, 34.0],  # Oklahoma
        'hugoton gas area': [-100.0, 36.0],  # Kansas/Oklahoma/Texas
        'woodford shale': [-96.0, 35.0],  # Oklahoma
        'utica shale': [-81.0, 40.0],  # Ohio/Pennsylvania
        'denver-julesburg basin': [-104.0, 40.0],  # Colorado
        'williston basin': [-102.0, 47.0],  # North Dakota
        'appalachian basin': [-80.0, 40.0],  # Pennsylvania/Ohio/West Virginia
        'gulf coast': [-95.0, 29.0],  # Texas/Louisiana
        'powder river basin': [-105.0, 44.0],  # Wyoming
        'san juan basin': [-108.0, 36.0],  # New Mexico/Colorado
        'piceance basin': [-108.0, 39.0],  # Colorado
        'green river basin': [-109.0, 41.0],  # Wyoming
        'wind river basin': [-108.0, 43.0],  # Wyoming
        'big horn basin': [-108.0, 44.0],  # Wyoming
        'raton basin': [-104.0, 37.0],  # Colorado/New Mexico
        'paradox basin': [-109.0, 38.0],  # Utah/Colorado
        'uinta basin': [-109.0, 40.0],  # Utah
        'piceance creek basin': [-108.0, 39.0],  # Colorado
        'denver basin': [-104.0, 40.0],  # Colorado
        'cheyenne basin': [-105.0, 41.0],  # Wyoming
        'laramie basin': [-105.0, 41.0],  # Wyoming
        'hanna basin': [-106.0, 42.0],  # Wyoming
        'great divide basin': [-108.0, 42.0],  # Wyoming
        'washakie basin': [-108.0, 43.0],  # Wyoming
        'bighorn basin': [-108.0, 44.0],  # Wyoming
        'powder river': [-105.0, 44.0],  # Wyoming
        'san juan': [-108.0, 36.0],  # New Mexico/Colorado
        'piceance': [-108.0, 39.0],  # Colorado
        'green river': [-109.0, 41.0],  # Wyoming
        'wind river': [-108.0, 43.0],  # Wyoming
        'big horn': [-108.0, 44.0],  # Wyoming
        'raton': [-104.0, 37.0],  # Colorado/New Mexico
        'paradox': [-109.0, 38.0],  # Utah/Colorado
        'uinta': [-109.0, 40.0],  # Utah
        'cheyenne': [-105.0, 41.0],  # Wyoming
        'laramie': [-105.0, 41.0],  # Wyoming
        'hanna': [-106.0, 42.0],  # Wyoming
        'great divide': [-108.0, 42.0],  # Wyoming
        'washakie': [-108.0, 43.0],  # Wyoming
        'bighorn': [-108.0, 44.0],  # Wyoming
    }
    
    # Check if it's a known US oil/gas region
    location_lower = location_text.lower().strip()
    if location_lower in us_oil_gas_regions:
        result = {
            "coordinates": us_oil_gas_regions[location_lower],
            "address": f"{location_text}, United States",
            "confidence": "high"
        }
        geocoding_cache.set(cache_key, result)  # Cache the result
        return result
    
    for attempt in range(max_retries):
        try:
            # Add a small delay to respect rate limits
            if attempt > 0:
                time.sleep(0.5)  # Reduced delay
            
            # For counties, try major oil/gas states first before generic USA
            if 'county' in location_text.lower():
                oil_gas_states = ['Oklahoma', 'Texas', 'North Dakota', 'New Mexico', 'Louisiana', 
                                 'Wyoming', 'Colorado', 'Pennsylvania', 'West Virginia']
                
                for state in oil_gas_states:
                    try:
                        location = geolocator.geocode(f"{location_text}, {state}, USA", timeout=5)  # Increased timeout
                        if location:
                            result = {
                                "coordinates": [location.longitude, location.latitude],
                                "address": location.address,
                                "confidence": "high"
                            }
                            geocoding_cache.set(cache_key, result)  # Cache the result
                            return result
                    except:
                        continue
            
            # Try geocoding with US bias
            location = geolocator.geocode(f"{location_text}, USA", timeout=5)  # Increased timeout
            
            if location:
                # Check if the result is in the US
                address = location.address.lower()
                if any(us_indicator in address for us_indicator in ['united states', 'usa', 'us']):
                    result = {
                        "coordinates": [location.longitude, location.latitude],
                        "address": location.address,
                        "confidence": "high"
                    }
                    geocoding_cache.set(cache_key, result)  # Cache the result
                    return result
            
            # If not in US, try without USA suffix
            location = geolocator.geocode(location_text, timeout=5)  # Increased timeout
            
            if location:
                # Check if the result is in the US
                address = location.address.lower()
                if any(us_indicator in address for us_indicator in ['united states', 'usa', 'us']):
                    result = {
                        "coordinates": [location.longitude, location.latitude],
                        "address": location.address,
                        "confidence": "medium"
                    }
                    geocoding_cache.set(cache_key, result)  # Cache the result
                    return result
            
            # If still not in US, try with "Texas, USA" suffix as a more common fallback
            location_with_state = f"{location_text}, Texas, USA"
            location = geolocator.geocode(location_with_state, timeout=5)  # Increased timeout
            
            if location:
                result = {
                    "coordinates": [location.longitude, location.latitude],
                    "address": location.address,
                    "confidence": "low"
                }
                geocoding_cache.set(cache_key, result)  # Cache the result
                return result
                
        except (GeocoderTimedOut, GeocoderServiceError) as e:
            print(f"Geocoding attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                return None
            time.sleep(1)  # Reduced wait time
        except Exception as e:
            print(f"Unexpected geocoding error: {e}")
            return None
    
    return None

# STR to Lat/Lng conversion function using geopandas and CRS transformations
def str_to_lat_lng(section, township, range_str):
    """
    Convert Section-Township-Range to lat/lng coordinates using proper geospatial calculations
    Uses geopandas and CRS transformations for accurate coordinate conversion
    """
    try:
        # Parse township and range
        township_num = int(township.replace('n', '').replace('s', '').replace('N', '').replace('S', ''))
        range_num = int(range_str.replace('e', '').replace('w', '').replace('E', '').replace('W', ''))
        section_num = int(section)
        
        # Determine direction multipliers
        township_dir = 1 if 'n' in township.lower() else -1
        range_dir = 1 if 'e' in range_str.lower() else -1
        
        # Oklahoma PLSS base coordinates (more accurate)
        # Base point for Oklahoma PLSS system - using actual PLSS origin
        base_lat = 35.0  # Base latitude for Oklahoma
        base_lng = -97.0  # Base longitude for Oklahoma
        
        # PLSS calculations with improved accuracy
        # Each township is 6 miles = 0.108 degrees (more accurate)
        # Each range is 6 miles = 0.108 degrees
        # Each section is 1 mile = 0.018 degrees
        
        # Calculate township and range offsets
        township_offset = (township_num - 8) * township_dir * 0.108  # 8 is base township
        range_offset = (range_num - 8) * range_dir * 0.108  # 8 is base range
        
        # Section offset within township (6x6 grid)
        section_row = (section_num - 1) // 6
        section_col = (section_num - 1) % 6
        section_lat_offset = section_row * 0.018
        section_lng_offset = section_col * 0.018
        
        # Calculate initial coordinates
        initial_lat = base_lat + township_offset + section_lat_offset
        initial_lng = base_lng + range_offset + section_lng_offset
        
        # Create a GeoDataFrame for CRS transformation
        point = Point(initial_lng, initial_lat)
        gdf = gpd.GeoDataFrame([1], geometry=[point], crs='EPSG:4326')
        
        # Transform to Oklahoma State Plane coordinate system for more accurate calculations
        # Oklahoma State Plane South (EPSG:2276)
        gdf_transformed = gdf.to_crs('EPSG:2276')
        
        # Get the transformed coordinates
        transformed_point = gdf_transformed.geometry.iloc[0]
        
        # Transform back to WGS84 (EPSG:4326) for final lat/lng
        gdf_final = gdf_transformed.to_crs('EPSG:4326')
        final_point = gdf_final.geometry.iloc[0]
        
        return [final_point.x, final_point.y]  # Return as [lng, lat]
        
    except Exception as e:
        print(f"Error converting STR to coordinates: {e}")
        # Fallback to simplified calculation
        try:
            township_num = int(township.replace('n', '').replace('s', '').replace('N', '').replace('S', ''))
            range_num = int(range_str.replace('e', '').replace('w', '').replace('E', '').replace('W', ''))
            section_num = int(section)
            
            township_dir = 1 if 'n' in township.lower() else -1
            range_dir = 1 if 'e' in range_str.lower() else -1
            
            # Simplified fallback calculation
            township_offset = (township_num - 8) * township_dir * 0.1
            range_offset = (range_num - 8) * range_dir * 0.1
            section_row = (section_num - 1) // 6
            section_col = (section_num - 1) % 6
            section_lat_offset = section_row * 0.0167
            section_lng_offset = section_col * 0.0167
            
            final_lat = 35.0 + township_offset + section_lat_offset
            final_lng = -97.0 + range_offset + section_lng_offset
            
            return [final_lng, final_lat]
        except:
            # Final fallback to Oklahoma City coordinates
            return [-97.5164, 35.4676]

# FastAPI setup
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", "http://localhost:3001", "http://localhost:3002", "http://localhost:3004", "http://localhost:3005",  # Local development
        "https://*.vercel.app",   # Vercel deployments
        "https://mineral-insights.vercel.app"  # Your production domain
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str
    conversation_id: str = "default"  # Default conversation ID

class PermitQueryRequest(BaseModel):
    query: str
    conversation_id: str = "default"
    use_database: bool = False  # True for SQLite, False for vector search

class MapQueryRequest(BaseModel):
    query: str
    conversation_id: str = "default"

class PDFUploadResponse(BaseModel):
    message: str
    filename: str
    pages_processed: int
    chunks_created: int

# Initialize Pinecone and OpenAI
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
embedder = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = PineconeVectorStore.from_existing_index(index_name="forum-db", embedding=embedder)

# Set up LLM with balanced temperature for focused but natural answers
llm = ChatOpenAI(model="gpt-4o", streaming=True, temperature=0.5)

# Initialize Cohere client for reranking
cohere_client = cohere.Client(os.getenv("COHERE_API_KEY"))

# Global conversation store
conversation_store: Dict[str, SimpleChatMessageHistory] = {}

# Create a prompt template with memory that distinguishes between permits and forum data
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a mineral rights expert. Use the context to give direct, specific answers.

Example:
Q: "What are lease rates in Grady County?"
Good: "Grady County rates are $500-$2,000/acre per forum discussions. Continental offers $1,500/acre with 22% royalty."
Bad: "Rates vary by location and many factors. Consult an attorney for your specific situation."

Rules:
1. Lead with specific numbers and facts from the context
2. Answer ONLY what was asked
3. Cite operators, counties, and dates mentioned in context
4. For lease rates: CRITICAL - Report what MOST people are being offered, not outliers
   - If you see 5 posts saying $1,000/acre and 1 post saying $10,000/acre, report $1,000/acre
   - Look for the TYPICAL or MEDIAN offer, not the highest or lowest
   - If there's a range, make it narrow around the most common values
   - Mention how many offers you found (e.g., "based on 8 recent offers")
# 5. Keep it concise and focused
5. Be conversational, follow up answers with a related question or comment

Context: {context}"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])

# Set up single retriever with k=30 to get more candidates for reranking
unified_retriever = vectorstore.as_retriever(search_kwargs={"k": 20})

def rerank_documents(query: str, documents: list, top_k: int = 11):
    """
    Rerank documents using Cohere's rerank-english-v3.0 model
    
    Args:
        query: The user's search query
        documents: List of Document objects from retriever
        top_k: Number of top documents to return after reranking
    
    Returns:
        List of reranked Document objects
    """
    if not documents:
        return []
    
    # Extract text content for reranking
    docs_text = [doc.page_content for doc in documents]
    
    try:
        # Call Cohere reranker
        results = cohere_client.rerank(
            model="rerank-english-v3.0",
            query=query,
            documents=docs_text,
            top_n=top_k
        )
        
        # Return reranked documents in order of relevance
        reranked_docs = [documents[r.index] for r in results.results]
        
        # Log reranking improvement
        print(f"🔍 Reranked {len(documents)} → {len(reranked_docs)} documents")
        if results.results:
            print(f"   Top score: {results.results[0].relevance_score:.3f}")
        
        return reranked_docs
    except Exception as e:
        print(f"⚠️  Reranking failed: {e}, returning original documents")
        # Fallback to original documents if reranking fails
        return documents[:top_k]

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_conversation_memory(conversation_id: str) -> SimpleChatMessageHistory:
    """Get or create conversation memory for a given conversation ID"""
    if conversation_id not in conversation_store:
        conversation_store[conversation_id] = SimpleChatMessageHistory()
    return conversation_store[conversation_id]

def get_recent_messages(conversation_id: str, k: int = 10) -> list:
    """Get the most recent k messages from conversation history"""
    memory = get_conversation_memory(conversation_id)
    messages = memory.messages
    # Return the last k messages (or all if less than k)
    return messages[-k:] if len(messages) > k else messages

# Create a memory-aware chain function that uses both permits and forum data
def create_memory_chain(conversation_id: str):
    """Create a RAG chain with conversation memory and reranking"""
    chat_history = get_recent_messages(conversation_id, k=10)
    
    def retrieve_and_rerank(query: str):
        """Retrieve documents and rerank them"""
        docs = unified_retriever.get_relevant_documents(query)
        reranked_docs = rerank_documents(query, docs, top_k=15)
        return format_docs(reranked_docs)
    
    chain = (
        {
            "context": RunnableLambda(retrieve_and_rerank),
            "question": RunnablePassthrough(),
            "chat_history": lambda x: chat_history
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

# Create streaming version with memory
def create_streaming_memory_chain(conversation_id: str):
    """Create a streaming RAG chain with conversation memory and reranking"""
    chat_history = get_recent_messages(conversation_id, k=10)
    
    def retrieve_and_rerank(query: str):
        """Retrieve documents and rerank them"""
        docs = unified_retriever.get_relevant_documents(query)
        reranked_docs = rerank_documents(query, docs, top_k=15)
        return format_docs(reranked_docs)
    
    chain = (
        {
            "context": RunnableLambda(retrieve_and_rerank),
            "question": RunnablePassthrough(),
            "chat_history": lambda x: chat_history
        }
        | prompt
        | llm
    )
    return chain

def extract_locations_with_ner(query):
    """
    Use spaCy NER to extract location entities from the query
    Returns a list of detected location names
    """
    if not SPACY_AVAILABLE:
        return []
    
    try:
        doc = nlp(query)
        locations = []
        
        # Common non-location words that spaCy might incorrectly identify as locations
        non_location_words = {
            'region', 'area', 'zone', 'district', 'sector', 'field', 'play', 'basin',
            'formation', 'shale', 'sand', 'zone', 'interval', 'horizon', 'reservoir',
            'deposit', 'occurrence', 'presence', 'activity', 'development', 'exploration',
            'production', 'drilling', 'completion', 'stimulation', 'fracturing'
        }
        
        # Common typos and corrections for oil & gas counties
        location_corrections = {
            'loen': 'leon',
            'leon county': 'leon county',
            'leon': 'leon',
            'dawson county': 'dawson county',
            'dawson': 'dawson',
            'grady county': 'grady county',
            'grady': 'grady',
            'washington county': 'washington county',
            'washington': 'washington',
            'freestone county': 'freestone county',
            'freestone': 'freestone',
            'robertson county': 'robertson county',
            'robertson': 'robertson',
            'rogers mills': 'roger mills',
            'roger mills county': 'roger mills county',
            'roger mills': 'roger mills'
        }
        
        # Extract GPE (Geo-Political Entities) and LOC (Locations)
        for ent in doc.ents:
            if ent.label_ in ["GPE", "LOC"]:
                # Filter out single-letter or very short entities
                if len(ent.text) >= 4:
                    # Filter out common non-location words
                    if ent.text.lower() not in non_location_words:
                        # Apply fuzzy matching corrections
                        corrected_text = ent.text.lower()
                        for typo, correction in location_corrections.items():
                            if typo in corrected_text:
                                corrected_text = corrected_text.replace(typo, correction)
                                print(f"🔧 Corrected '{ent.text}' to '{corrected_text}'")
                                break
                        
                        locations.append(corrected_text.title())
                        print(f"🔍 spaCy detected location: '{ent.text}' -> '{corrected_text.title()}' (type: {ent.label_})")
                    else:
                        print(f"🔍 spaCy detected but filtered out: '{ent.text}' (common non-location word)")
        
        return locations
    except Exception as e:
        print(f"Error in spaCy NER: {e}")
        return []

# Parse location data from query using dynamic geocoding
def extract_location(query):
    query_lower = query.lower()
    
    # OIL & GAS STATE WEIGHTING: Heavy preference for states with oil & gas activity mentioned in forum data
    # Based on your mineral rights forum data, these are the states with significant activity
    oil_gas_states = ['texas', 'oklahoma', 'louisiana', 'north dakota', 'colorado', 'pennsylvania', 'ohio', 'west virginia', 'wyoming', 'new mexico', 'kansas', 'arkansas']
    
    # Clarification feature removed - process county-only queries normally
    
    # First, check for Section-Township-Range patterns (keep this as it's specific to mineral rights)
    str_patterns = [
        r"(?:sec|section)\s+(\d{1,2})\s+(\d{1,2}[ns])\s+(\d{1,2}[ew])",
        r"(\d{1,2})\s+(\d{1,2}[ns])\s+(\d{1,2}[ew])",
        r"section\s+(\d{1,2})\s+township\s+(\d{1,2}[ns])\s+range\s+(\d{1,2}[ew])"
    ]
    
    for pattern in str_patterns:
        str_match = re.search(pattern, query_lower)
        if str_match:
            section = str_match.group(1)
            township = str_match.group(2)
            range_str = str_match.group(3)
            
            # Convert STR to coordinates using the geospatial function
            coordinates = str_to_lat_lng(section, township, range_str)
            
            return {
                "type": "str",
                "section": section,
                "township": township,
                "range": range_str,
                "coordinates": coordinates,
                "zoom": 12
            }
    
    # Check if this query is even asking about a location
    # Skip geocoding for queries that are clearly not about locations
    non_location_keywords = [
        'how do', 'what is', 'what are', 'how does', 'explain', 'tell me about',
        'worth', 'value', 'price', 'cost', 'rate', 'lease', 'agreement', 'contract',
        'rights', 'ownership', 'legal', 'law', 'regulation', 'protect', 'mineral',
        'what about', 'the region', 'the area', 'the zone', 'the field', 'the play'
    ]
    
    # If the query contains these phrases and NO clear location indicators, skip geocoding
    has_non_location_keywords = any(keyword in query_lower for keyword in non_location_keywords)
    
    # Strong location indicators that suggest the user wants to see a location
    strong_location_indicators = [
        'show me', 'zoom to', 'go to', 'find', 'locate', 'where is', 'map',
        'county', 'city', 'town', 'basin', 'formation', 'shale', 'field'
    ]
    
    has_strong_location_indicator = any(indicator in query_lower for indicator in strong_location_indicators)
    
    # If query has non-location keywords and no strong location indicators, skip geocoding
    if has_non_location_keywords and not has_strong_location_indicator:
        return None
    
    # Extract potential location terms from the query
    location_terms = []
    
    # FIRST: Try spaCy NER to get high-confidence location entities
    if SPACY_AVAILABLE:
        ner_locations = extract_locations_with_ner(query)
        location_terms.extend(ner_locations)
        
        # If spaCy found clear locations, prioritize those
        if ner_locations:
            print(f"✅ Using spaCy NER results: {ner_locations}")
    
    # Common location indicators (fallback if NER doesn't find anything)
    location_indicators = [
        'in', 'at', 'near', 'around', 'show me', 'zoom to', 'go to', 'find',
        'county', 'city', 'town', 'state', 'region', 'basin', 'formation'
    ]
    
    # First, try to extract complete location phrases
    # Look for patterns like "County, State" or "City, State"
    
    # Pattern for "County, State" or "City, State" - look for county/city/town specifically
    county_city_pattern = r'\b([a-zA-Z]+(?:\s+[a-zA-Z]+){0,2}\s+(?:county|city|town)),?\s*([a-zA-Z]{2,})\b'
    matches = re.findall(county_city_pattern, query_lower)
    
    # If the above pattern captures too much, try a more targeted approach
    if matches and len(matches[0][0]) > 20:  # If captured text is too long
        # Look for county/city/town more specifically
        simple_county_pattern = r'\b([a-zA-Z]+\s+county),?\s*([a-zA-Z]{2,})\b'
        simple_matches = re.findall(simple_county_pattern, query_lower)
        if simple_matches:
            matches = simple_matches
    
    for match in matches:
        location_name = match[0].strip()
        state = match[1].strip()
        
        # Clean up location name - remove common prefixes
        location_name = re.sub(r'^(show me|tell me about|drilling activity in|what about|lease rate for|in)\s+', '', location_name, flags=re.IGNORECASE)
        # Convert state abbreviation to full name if needed
        state_abbrev_map = {
            'tx': 'texas', 'ca': 'california', 'ny': 'new york', 'fl': 'florida',
            'il': 'illinois', 'pa': 'pennsylvania', 'oh': 'ohio', 'ga': 'georgia',
            'nc': 'north carolina', 'mi': 'michigan', 'nj': 'new jersey', 'va': 'virginia',
            'wa': 'washington', 'az': 'arizona', 'ma': 'massachusetts', 'tn': 'tennessee',
            'in': 'indiana', 'mo': 'missouri', 'md': 'maryland', 'wi': 'wisconsin',
            'co': 'colorado', 'mn': 'minnesota', 'sc': 'south carolina', 'al': 'alabama',
            'la': 'louisiana', 'ky': 'kentucky', 'or': 'oregon', 'ok': 'oklahoma',
            'ct': 'connecticut', 'ut': 'utah', 'ia': 'iowa', 'nv': 'nevada',
            'ar': 'arkansas', 'ms': 'mississippi', 'ks': 'kansas', 'nm': 'new mexico',
            'ne': 'nebraska', 'wv': 'west virginia', 'id': 'idaho', 'hi': 'hawaii',
            'nh': 'new hampshire', 'me': 'maine', 'mt': 'montana', 'ri': 'rhode island',
            'de': 'delaware', 'sd': 'south dakota', 'nd': 'north dakota', 'ak': 'alaska',
            'vt': 'vermont', 'wy': 'wyoming'
        }
        full_state = state_abbrev_map.get(state.lower(), state)
        full_location = f"{location_name}, {full_state}"
        
        # Heavy weighting for oil & gas states - prioritize these locations
        oil_gas_boost = 2.0 if full_state.lower() in oil_gas_states else 1.0
        print(f"🎯 Oil & gas state weighting: {oil_gas_boost}x for '{full_state}'")
            
        # Try geocoding this county/state combination immediately
        print(f"Attempting to geocode county/state: '{full_location}'")
        geocode_result = geocode_location(full_location)
        
        if geocode_result:
            return {
                "type": "county",
                "name": full_location,
                "coordinates": geocode_result["coordinates"],
                "zoom": 9,
                "address": geocode_result.get("address", ""),
                "confidence": geocode_result.get("confidence", "high")
            }
        
        location_terms.append(full_location)
    
    # Extract standalone county names (e.g., "karnes county", "reeves county")
    # This handles cases where county is mentioned without a state
    # Match one or two words followed by the word "county", excluding common prepositions
    standalone_county_pattern = r'(?:^|(?<=\s))([a-z]+(?:\s+[a-z]+)?\s+county)(?=\s|$|[,.\?!])'
    standalone_matches = re.findall(standalone_county_pattern, query_lower)
    # Filter out matches that start with prepositions
    prepositions = ['in', 'at', 'to', 'from', 'on', 'for', 'with', 'about']
    standalone_matches = [m for m in standalone_matches if not any(m.startswith(prep + ' ') for prep in prepositions)]
    
    # Also filter out if the county is followed by a state name (already handled above)
    # Check if there's ", texas" or ", oklahoma" etc after the county
    state_names = ['texas', 'oklahoma', 'louisiana', 'arkansas', 'kansas', 'new mexico', 
                   'wyoming', 'colorado', 'pennsylvania', 'ohio', 'west virginia', 'north dakota']
    filtered_standalone = []
    for county in standalone_matches:
        # Check if this county is followed by a state in the query
        county_pos = query_lower.find(county)
        if county_pos != -1:
            after_county = query_lower[county_pos + len(county):county_pos + len(county) + 30]
            # If followed by comma and state name, skip it (already handled)
            if not any(f', {state}' in after_county or f' {state}' in after_county for state in state_names):
                filtered_standalone.append(county)
        else:
            filtered_standalone.append(county)
    
    if filtered_standalone:
        for county_name in filtered_standalone:
            # Clean up and add to location terms
            clean_county = county_name.replace(' co', ' county')
            location_terms.append(clean_county)
            print(f"Found standalone county: '{clean_county}'")
    
    # Split query into words and look for location terms
    words = query_lower.split()
    
    # Find potential location phrases
    for i, word in enumerate(words):
        # Check if word is a location indicator
        if word in location_indicators:
            # Look for the next few words as potential location
            if i + 1 < len(words):
                # Try 1-4 word combinations after the indicator
                for j in range(1, min(5, len(words) - i)):
                    location_phrase = ' '.join(words[i+1:i+1+j])
                    if len(location_phrase) > 2:  # Avoid very short phrases
                        location_terms.append(location_phrase)
        
        # Also check for standalone location-like words (longer than 5 chars)na
        # But exclude common non-location words
        excluded_words = {
            'what', 'where', 'when', 'how', 'why', 'tell', 'about', 'mineral', 'rights', 
            'oil', 'gas', 'lease', 'valuation', 'drilling', 'activity', 'going', 'rate',
            'land', 'for', 'with', 'the', 'this', 'that', 'these', 'those', 'and', 'or',
            'but', 'if', 'then', 'because', 'since', 'until', 'while', 'during', 'before',
            'after', 'above', 'below', 'between', 'among', 'through', 'across', 'around',
            'behind', 'beside', 'inside', 'outside', 'within', 'without', 'under', 'over',
            'there', 'their', 'they', 'them', 'area', 'from', 'into', 'onto', 'upon',
            'lease', 'leases', 'current', 'market', 'rates', 'western', 'eastern', 'northern',
            'southern', 'price', 'prices', 'worth', 'value', 'legal', 'protection', 'work'
        }
        # Increase minimum length to 6 characters and only accept capitalized proper nouns
        if len(word) > 5 and word not in excluded_words and word.isalpha():
            # ONLY add words that are capitalized (proper nouns) AND not common words
            if word[0].isupper() and word.lower() not in excluded_words:
                location_terms.append(word)
    
    # REMOVED: Don't add the entire query as a fallback - too often incorrect
    
    # Check for oil and gas regions FIRST (before individual word extraction)
    oil_gas_keywords = ['basin', 'formation', 'shale', 'field', 'play', 'reservoir']
    if any(keyword in query_lower for keyword in oil_gas_keywords):
        print(f"Attempting to geocode oil/gas region: '{query}'")
        geocode_result = geocode_location(query)
        
        if geocode_result:
            return {
                "type": "region",
                "name": query,
                "coordinates": geocode_result["coordinates"],
                "zoom": 8,
                "address": geocode_result.get("address", ""),
                "confidence": geocode_result.get("confidence", "medium")
            }
    
    # Oil & gas states already defined at top of function
    
    # Try to geocode each potential location term (but be selective)
    for term in location_terms:
        # Skip very short terms (less than 4 chars)
        if len(term.strip()) < 4:
            continue
            
        # Skip if the term is just a common word
        if term.lower() in excluded_words:
            continue
        
        # Check if this location is in an oil & gas state before geocoding
        # This prevents zooming to non-oil & gas areas
        print(f"Attempting to geocode: '{term}'")
        geocode_result = geocode_location(term)
        
        # If geocoding succeeded, check if it's in an oil & gas state
        if geocode_result:
            address = geocode_result.get("address", "").lower()
            is_oil_gas_state = any(state in address for state in oil_gas_states)
            oil_gas_boost = 2.0 if is_oil_gas_state else 1.0
            print(f"🎯 Oil & gas state weighting: {oil_gas_boost}x for '{term}' -> {address}")
        
        if geocode_result:
            # Determine location type and zoom level based on the result
            location_type = "location"
            zoom_level = 10
            
            # Try to determine type from the address
            address = geocode_result.get("address", "").lower()
            if "county" in address:
                location_type = "county"
                zoom_level = 9
            elif any(city_word in address for city_word in ["city", "town", "village"]):
                location_type = "city"
                zoom_level = 10
            elif "state" in address or "oklahoma" in address:
                location_type = "state"
                zoom_level = 7
            elif any(region_word in address for region_word in ["region", "area", "basin", "formation"]):
                location_type = "region"
                zoom_level = 8
            
            return {
                "type": location_type,
                "name": term,
                "coordinates": geocode_result["coordinates"],
                "zoom": zoom_level,
                "address": geocode_result.get("address", ""),
                "confidence": geocode_result.get("confidence", "medium")
            }
    
    # No location found - DO NOT try geocoding the entire query as it causes false positives
    return None

def query_permits_database(query: str) -> List[Dict]:
    """Query the SQLite permits database for structured data"""
    try:
        db_path = "Drilling Permits/data/permits.db"
        if not Path(db_path).exists():
            return []
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Simple keyword search across key fields
        search_terms = query.lower().split()
        conditions = []
        params = []
        
        for term in search_terms:
            conditions.append("""
                (LOWER(entity_name) LIKE ? OR 
                 LOWER(county) LIKE ? OR 
                 LOWER(well_name) LIKE ? OR 
                 LOWER(formation_name) LIKE ? OR
                 LOWER(permit_status) LIKE ?)
            """)
            params.extend([f"%{term}%"] * 5)
        
        if conditions:
            sql = f"""
                SELECT api_number, entity_name, well_name, well_type, county, 
                       section, township, range, approval_date, permit_status, remarks
                FROM permits 
                WHERE {' AND '.join(conditions)}
                ORDER BY approval_date DESC
                LIMIT 10
            """
            
            cursor.execute(sql, params)
            results = cursor.fetchall()
            
            permits = []
            for row in results:
                permits.append({
                    'api_number': row[0],
                    'operator': row[1],
                    'well_name': row[2],
                    'well_type': row[3],
                    'county': row[4],
                    'section': row[5],
                    'township': row[6],
                    'range': row[7],
                    'approval_date': row[8],
                    'permit_status': row[9],
                    'remarks': row[10]
                })
            
            conn.close()
            return permits
        
    except Exception as e:
        print(f"Database query error: {e}")
    
    return []

# PDF Processing Functions
async def process_pdf_content(pdf_content: bytes, filename: str) -> dict:
    """Process PDF content and add to vector database"""
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
            temp_file.write(pdf_content)
            temp_file_path = temp_file.name
        
        # Read PDF content
        pdf_reader = PdfReader(temp_file_path)
        text_content = ""
        pages_processed = 0
        
        for page in pdf_reader.pages:
            text_content += page.extract_text() + "\n"
            pages_processed += 1
        
        # Clean up temporary file
        os.unlink(temp_file_path)
        
        if not text_content.strip():
            return {
                "success": False,
                "message": "No text content found in PDF",
                "pages_processed": pages_processed,
                "chunks_created": 0
            }
        
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        chunks = text_splitter.split_text(text_content)
        
        # Add metadata to each chunk
        documents_with_metadata = []
        for i, chunk in enumerate(chunks):
            documents_with_metadata.append({
                "content": chunk,
                "metadata": {
                    "source": filename,
                    "page": (i // 10) + 1,  # Approximate page number
                    "chunk_id": i,
                    "type": "pdf_upload"
                }
            })
        
        # Add to vector database
        if documents_with_metadata:
            vectorstore.add_texts(
                texts=[doc["content"] for doc in documents_with_metadata],
                metadatas=[doc["metadata"] for doc in documents_with_metadata]
            )
        
        return {
            "success": True,
            "message": f"Successfully processed PDF: {filename}",
            "pages_processed": pages_processed,
            "chunks_created": len(chunks)
        }
        
    except Exception as e:
        return {
            "success": False,
            "message": f"Error processing PDF: {str(e)}",
            "pages_processed": 0,
            "chunks_created": 0
        }

# API endpoint for PDF upload
@app.post("/upload-pdf", response_model=PDFUploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """Upload and process a PDF file for RAG"""
    try:
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        
        # Check file size (limit to 10MB)
        file_content = await file.read()
        if len(file_content) > 10 * 1024 * 1024:  # 10MB
            raise HTTPException(status_code=400, detail="File size too large. Maximum 10MB allowed.")
        
        # Process the PDF
        result = await process_pdf_content(file_content, file.filename)
        
        if result["success"]:
            return PDFUploadResponse(
                message=result["message"],
                filename=file.filename,
                pages_processed=result["pages_processed"],
                chunks_created=result["chunks_created"]
            )
        else:
            raise HTTPException(status_code=400, detail=result["message"])
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading PDF: {str(e)}")

# API endpoint for chat
@app.post("/chat")
async def chat(query: QueryRequest):
    try:
        location_data = extract_location(query.query)
        
        # Clarification feature removed
        
        # Get conversation history for LangGraph
        memory = get_conversation_memory(query.conversation_id)
        conversation_history = memory.messages
        
        # Get response using LangGraph
        result = run_mineral_query(query.query, conversation_history)
        
        # Check if this is a mapping query and get mapping data
        mapping_data = None
        mapping_summary = None
        needs_mapping = False
        
        query_lower = query.query.lower()
        mapping_keywords = ["map", "show me", "display", "visualize", "plot", "location", "where are", "drilling permits", "wells"]
        is_mapping_query = any(keyword in query_lower for keyword in mapping_keywords)
        
        if is_mapping_query:
            try:
                mapper = DrillingPermitsMapper()
                permits_data = mapper.process_query(query.query)
                if permits_data and permits_data.get("geojson", {}).get("features"):
                    mapping_data = permits_data["geojson"]
                    mapping_summary = permits_data.get("summary", {})
                    needs_mapping = True
                    print(f"🗺️ Mapping data found: {len(mapping_data['features'])} permits")
            except Exception as e:
                print(f"⚠️ Mapping error: {e}")
        
        # Save conversation to memory
        memory.add_user_message(query.query)
        memory.add_ai_message(result["answer"])
        
        return {
            "answer": result["answer"],
            "location": location_data,
            "conversation_id": query.conversation_id,
            "confidence": result["confidence"],
            "sources": result["sources"],
            "needs_mapping": needs_mapping,
            "mapping_data": mapping_data,
            "mapping_summary": mapping_summary
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Permits-specific endpoint
@app.post("/chat/permits")
async def permits_chat(query: PermitQueryRequest):
    try:
        if query.use_database:
            # Use SQLite database for structured queries
            permits_data = query_permits_database(query.query)
            
            if permits_data:
                # Format permits data for LLM
                permits_text = "\n\n".join([
                    f"Permit: {p['api_number']}\n"
                    f"Operator: {p['operator']}\n"
                    f"Well: {p['well_name']}\n"
                    f"Location: {p['county']} County, Section {p['section']}, {p['township']}, {p['range']}\n"
                    f"Status: {p['permit_status']}\n"
                    f"Approved: {p['approval_date']}\n"
                    f"Notes: {p['remarks']}"
                    for p in permits_data
                ])
                
                # Create a simple chain for permits-only queries
                permits_chain = (
                    {"context": lambda x: permits_text, "question": RunnablePassthrough()}
                    | ChatPromptTemplate.from_template("""
                        Based on the following Oklahoma drilling permits data, answer the user's question:
                        
                        Permits Data:
                        {context}
                        
                        Question: {question}
                        
                        Provide a clear, helpful answer about the drilling permits. Include specific details like operator names, well names, locations, and approval dates when relevant.
                    """)
                    | llm
                    | StrOutputParser()
                )
                
                response = permits_chain.invoke(query.query)
                
            else:
                response = "No permits found matching your search criteria. The permits database contains recent drilling permits from Oklahoma (last 6 months)."
                
        else:
            # Use vector search (permits-only)
            permits_chain = (
                {"context": permits_retriever | format_docs, "question": RunnablePassthrough()}
                | ChatPromptTemplate.from_template("""
                    Based on the following Oklahoma drilling permits information, answer the user's question:
                    
                    Permits Context: {context}
                    
                    Question: {question}
                    
                    Provide a clear, helpful answer about the drilling permits. Include specific details when relevant.
                """)
                | llm
                | StrOutputParser()
            )
            
            response = permits_chain.invoke(query.query)
        
        # Save to conversation memory
        memory = get_conversation_memory(query.conversation_id)
        memory.add_user_message(query.query)
        memory.add_ai_message(response)
        
        return {
            "answer": response,
            "conversation_id": query.conversation_id,
            "data_source": "sqlite_database" if query.use_database else "vector_search",
            "query_type": "permits_only"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Streaming chat endpoint
@app.post("/chat/stream")
async def chat_stream(query: QueryRequest):
    async def generate_stream():
        try:
            # Get location data first (non-streaming)
            location_data = extract_location(query.query)
            
            # Clarification feature removed
            
            # Send initial response with location data
            initial_response = {
                "type": "location",
                "data": location_data
            }
            yield f"data: {json.dumps(initial_response)}\n\n"
            
            # Get conversation history for LangGraph
            memory = get_conversation_memory(query.conversation_id)
            conversation_history = memory.messages
            
            # Stream the LangGraph response
            try:
                full_response = ""
                async for chunk in stream_mineral_query(query.query, conversation_history):
                    full_response += chunk
                    
                    stream_response = {
                        "type": "content",
                        "data": chunk
                    }
                    yield f"data: {json.dumps(stream_response)}\n\n"
                
                # Save conversation to memory after streaming completes
                memory.add_user_message(query.query)
                memory.add_ai_message(full_response)
                    
            except Exception as stream_error:
                # Send error response
                error_response = {
                    "type": "error",
                    "data": f"Streaming error: {str(stream_error)}"
                }
                yield f"data: {json.dumps(error_response)}\n\n"
            
            # Check if this was a mapping query and include mapping data
            mapping_data = None
            mapping_summary = None
            
            # Simple mapping detection for streaming
            query_lower = query.query.lower()
            mapping_keywords = ["map", "show me", "display", "visualize", "plot", "location", "where are", "drilling permits", "wells"]
            is_mapping_query = any(keyword in query_lower for keyword in mapping_keywords)
            
            if is_mapping_query:
                try:
                    from mapping_agent import DrillingPermitsMapper
                    mapper = DrillingPermitsMapper()
                    permits_data = mapper.process_query(query.query)
                    if permits_data and permits_data.get("geojson", {}).get("features"):
                        mapping_data = permits_data["geojson"]
                        mapping_summary = permits_data.get("summary", {})
                except Exception as e:
                    print(f"⚠️ Mapping error in stream: {e}")
            
            # Send completion signal
            completion_response = {
                "type": "done",
                "data": "Stream completed",
                "mapping_data": mapping_data,
                "mapping_summary": mapping_summary
            }
            yield f"data: {json.dumps(completion_response)}\n\n"
            
        except Exception as e:
            # Send error response
            error_response = {
                "type": "error",
                "data": str(e)
            }
            yield f"data: {json.dumps(error_response)}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/plain; charset=utf-8"
        }
    )

# API endpoint to get conversation history
@app.get("/conversation/{conversation_id}")
async def get_conversation_history(conversation_id: str):
    try:
        memory = get_conversation_memory(conversation_id)
        messages = memory.messages
        
        # Format messages for API response
        formatted_messages = []
        for msg in messages:
            formatted_messages.append({
                "type": "human" if isinstance(msg, HumanMessage) else "ai",
                "content": msg.content,
                "timestamp": getattr(msg, 'timestamp', None)
            })
        
        return {
            "conversation_id": conversation_id,
            "message_count": len(messages),
            "messages": formatted_messages
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# API endpoint to clear conversation history
@app.delete("/conversation/{conversation_id}")
async def clear_conversation_history(conversation_id: str):
    try:
        if conversation_id in conversation_store:
            del conversation_store[conversation_id]
            return {"message": f"Conversation {conversation_id} cleared successfully"}
        else:
            return {"message": f"Conversation {conversation_id} not found"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Database stats endpoint
@app.get("/permits/stats")
async def get_permits_stats():
    """Get statistics about the permits database"""
    try:
        db_path = "Drilling Permits/data/permits.db"
        if not Path(db_path).exists():
            return {"error": "Permits database not found"}
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get basic stats
        cursor.execute("SELECT COUNT(*) FROM permits")
        total_permits = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT entity_name) FROM permits")
        unique_operators = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT county) FROM permits")
        unique_counties = cursor.fetchone()[0]
        
        cursor.execute("SELECT MIN(approval_date), MAX(approval_date) FROM permits WHERE approval_date != ''")
        date_range = cursor.fetchone()
        
        conn.close()
        
        return {
            "total_permits": total_permits,
            "unique_operators": unique_operators,
            "unique_counties": unique_counties,
            "date_range": {
                "earliest": date_range[0],
                "latest": date_range[1]
            },
            "database_path": db_path
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Mapping agent endpoint
@app.post("/map/permits")
async def map_permits(query: MapQueryRequest):
    """Map drilling permits based on location query"""
    try:
        # Initialize mapping agent
        mapper = DrillingPermitsMapper()
        
        # Process the query
        result = mapper.process_query(query.query)
        
        # Save to conversation memory
        memory = get_conversation_memory(query.conversation_id)
        memory.add_user_message(f"Map query: {query.query}")
        memory.add_ai_message(f"Found {result['summary']['total_permits']} permits for mapping")
        
        return {
            "geojson": result["geojson"],
            "bounds": result["bounds"],
            "summary": result["summary"],
            "location_params": result["location_params"],
            "conversation_id": query.conversation_id,
            "query": query.query
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# API endpoint to serve wells GeoJSON data (filtered - no plugged/abandoned wells)
@app.get("/wells")
async def get_wells():
    try:
        # For Vercel deployment, we'll need to handle large files differently
        # Check if we're in a Vercel environment
        if os.getenv("VERCEL"):
            # Try to serve the Vercel-optimized dataset first (500 wells)
            vercel_file = "RBDMS_WELLS_VERCEL.geojson"
            if os.path.exists(vercel_file):
                return FileResponse(
                    vercel_file,
                    media_type="application/json",
                    filename="rbdms_wells_vercel.geojson"
                )
            # Fallback to sample dataset if available
            sample_file = "RBDMS_WELLS_SAMPLE.geojson"
            if os.path.exists(sample_file):
                return FileResponse(
                    sample_file,
                    media_type="application/json",
                    filename="rbdms_wells_sample.geojson"
                )
            else:
                # Fallback: return a message directing to use the count endpoint
                return JSONResponse({
                    "message": "Large dataset not available in serverless environment",
                    "suggestion": "Use /wells/count for statistics or host data on CDN",
                    "count": 176824,
                    "note": "For full dataset access, consider using AWS S3 or similar CDN"
                })
        
        # Local development - serve the file directly
        # Try Vercel-optimized file first (500 wells)
        vercel_file = "RBDMS_WELLS_VERCEL.geojson"
        if os.path.exists(vercel_file):
            return FileResponse(
                vercel_file,
                media_type="application/json",
                filename="rbdms_wells_vercel.geojson"
            )
        # Fallback to filtered file
        wells_file = "RBDMS_WELLS_FILTERED.geojson"
        if os.path.exists(wells_file):
            return FileResponse(
                wells_file,
                media_type="application/json",
                filename="rbdms_wells_filtered.geojson"
            )
        else:
            # Fallback to original file if filtered doesn't exist
            original_file = "RBDMS_WELLS.geojson"
            if os.path.exists(original_file):
                return FileResponse(
                    original_file,
                    media_type="application/json",
                    filename="rbdms_wells.geojson"
                )
            else:
                raise HTTPException(status_code=404, detail="Wells data file not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# API endpoint to get wells count
@app.get("/wells/count")
async def get_wells_count():
    try:
        # For Vercel deployment, return cached count
        if os.getenv("VERCEL"):
            return {"count": 500, "note": "Vercel-optimized dataset (500 wells)"}
        
        # Local development - read from file
        import json
        # Check Vercel-optimized file first
        vercel_file = "RBDMS_WELLS_VERCEL.geojson"
        if os.path.exists(vercel_file):
            with open(vercel_file, 'r') as f:
                data = json.load(f)
                return {"count": len(data.get("features", [])), "note": "Vercel-optimized dataset"}
        
        wells_file = "RBDMS_WELLS_FILTERED.geojson"
        if not os.path.exists(wells_file):
            # Fallback to original file
            wells_file = "RBDMS_WELLS.geojson"
        
        if os.path.exists(wells_file):
            with open(wells_file, 'r') as f:
                data = json.load(f)
                return {"count": len(data.get("features", []))}
        else:
            # Fallback to cached count if file not found
            return {"count": 176824, "note": "Using cached count - file not found"}
    except Exception as e:
        # Fallback to cached count on error
        return {"count": 176824, "note": "Using cached count due to error", "error": str(e)}

# API endpoint to get wells for a specific bounding box (viewport-based loading)
@app.get("/wells/bbox")
async def get_wells_bbox(minx: float, miny: float, maxx: float, maxy: float, limit: int = 1000):
    try:
        import json
        import geopandas as gpd
        from shapely.geometry import box
        
        wells_file = "/Users/fmt116/Desktop/Mineral Insights/RBDMS_WELLS.geojson"
        if not os.path.exists(wells_file):
            raise HTTPException(status_code=404, detail="Wells data file not found")
        
        # Create bounding box
        bbox = box(minx, miny, maxx, maxy)
        
        # Read only the first part of the file to get structure
        with open(wells_file, 'r') as f:
            # Read in chunks to avoid loading entire file
            chunk_size = 1024 * 1024  # 1MB chunks
            content = ""
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                content += chunk
                # Stop after we have enough data to work with
                if len(content) > 50 * 1024 * 1024:  # 50MB limit for processing
                    break
        
        # For now, return a simplified response indicating too many wells
        return {
            "message": "Too many wells for real-time loading",
            "total_wells": 452086,
            "suggestion": "Use clustering or sampling for better performance"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# API endpoint to get drilling permits by location
@app.get("/permits/location")
async def get_permits_by_location(
    county: str = None,
    lat: float = None,
    lng: float = None,
    section: str = None,
    township: str = None,
    range_str: str = None,
    radius_miles: float = 50,
    limit: int = 100
):
    """
    Get drilling permits for a specific location.
    Can search by:
    - County name
    - Lat/lng coordinates
    - Section-Township-Range (STR)
    
    Query params:
    - county: County name (e.g., "KARNES", "Karnes County")
    - lat, lng: Coordinates for radius search
    - section, township, range_str: Section-Township-Range (e.g., section=32, township=8N, range_str=8W)
    - radius_miles: Search radius in miles (default 50 for coords, 10 for STR)
    - limit: Max permits to return (default 100)
    """
    try:
        import pandas as pd
        from math import radians, cos, sin, asin, sqrt
        
        def haversine(lon1, lat1, lon2, lat2):
            """Calculate distance between two points in miles"""
            lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
            dlon = lon2 - lon1
            dlat = lat2 - lat1
            a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
            c = 2 * asin(sqrt(a))
            miles = 3956 * c  # Radius of earth in miles
            return miles
        
        permits = []
        
        # If STR is provided, convert to lat/lng first
        if section and township and range_str:
            str_coords = str_to_lat_lng(section, township, range_str)
            if str_coords:
                lat, lng = str_coords
                # Use smaller default radius for STR queries (more precise)
                if radius_miles == 50:  # If using default
                    radius_miles = 10
                print(f"Converted STR (Sec {section}, T{township}, R{range_str}) to coordinates: {lat}, {lng}")
            else:
                raise HTTPException(status_code=400, detail="Invalid Section-Township-Range values")
        
        # Try to load from SQLite database first
        db_path = "Drilling Permits/data/permits.db"
        if os.path.exists(db_path):
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            if county:
                county_clean = county.upper().replace(" COUNTY", "").replace(" CO", "").strip()
                cursor.execute("""
                    SELECT api_number, entity_name, well_name, county, state, 
                           surf_lat_y, surf_long_x, formation_name, total_depth, 
                           well_type, permit_type, well_status
                    FROM permits 
                    WHERE UPPER(county) LIKE ? 
                    LIMIT ?
                """, (f"%{county_clean}%", limit))
            elif lat is not None and lng is not None:
                cursor.execute("""
                    SELECT api_number, entity_name, well_name, county, state, 
                           surf_lat_y, surf_long_x, formation_name, total_depth, 
                           well_type, permit_type, well_status
                    FROM permits 
                    WHERE surf_lat_y IS NOT NULL AND surf_long_x IS NOT NULL
                    LIMIT ?
                """, (limit,))
            else:
                raise HTTPException(status_code=400, detail="Must provide either county or lat/lng")
            
            rows = cursor.fetchall()
            
            for row in rows:
                permit = {
                    "api_number": str(row[0]) if row[0] else "",
                    "operator": str(row[1]) if row[1] else "",
                    "well_name": str(row[2]) if row[2] else "",
                    "county": str(row[3]) if row[3] else "",
                    "state": str(row[4]) if row[4] else "",
                    "latitude": float(row[5]) if row[5] is not None else None,
                    "longitude": float(row[6]) if row[6] is not None else None,
                    "formation": str(row[7]) if row[7] else "",
                    "depth": float(row[8]) if row[8] is not None else None,
                    "well_type": str(row[9]) if row[9] else "",
                    "permit_date": str(row[10]) if row[10] else "",
                    "status": str(row[11]) if row[11] else ""
                }
                permits.append(permit)
            
            conn.close()
        elif os.path.exists("Texas Drilling Permits/data/texas_permits_20251004.csv"):
            df_tx = pd.read_csv(tx_file)
            
            if county:
                # Clean county name for matching (remove spaces for matching multi-word counties like "DE WITT")
                county_clean = county.upper().replace(" COUNTY", "").replace(" CO", "").replace(" ", "").strip()
                # Filter by county (also remove spaces from CSV county names for matching)
                df_filtered = df_tx[df_tx['County'].str.upper().str.replace(" ", "").str.contains(county_clean, na=False)]
            elif lat is not None and lng is not None:
                # Filter by distance (if coordinates available in permits)
                df_filtered = df_tx.copy()
                # Add distance column if lat/lng available
                if 'Latitude' in df_tx.columns and 'Longitude' in df_tx.columns:
                    df_filtered = df_filtered.dropna(subset=['Latitude', 'Longitude'])
                    df_filtered['distance'] = df_filtered.apply(
                        lambda row: haversine(lng, lat, float(row['Longitude']), float(row['Latitude'])),
                        axis=1
                    )
                    df_filtered = df_filtered[df_filtered['distance'] <= radius_miles]
                    df_filtered = df_filtered.sort_values('distance')
            else:
                raise HTTPException(status_code=400, detail="Must provide either county or lat/lng")
            
            # Limit results
            df_filtered = df_filtered.head(limit)
            
            # Convert to dict
            for _, row in df_filtered.iterrows():
                # Get lat/lng - if not available, try to convert from State Plane X/Y
                latitude = None
                longitude = None
                
                if pd.notna(row.get('Latitude')) and pd.notna(row.get('Longitude')):
                    latitude = float(row['Latitude'])
                    longitude = float(row['Longitude'])
                elif pd.notna(row.get('X_Coordinate')) and pd.notna(row.get('Y_Coordinate')):
                    # Convert State Plane to Lat/Long
                    try:
                        from pyproj import Transformer
                        x = float(row['X_Coordinate'])
                        y = float(row['Y_Coordinate'])
                        zone = str(row.get('State_Plane_Zone', 'Central')).strip()
                        datum = str(row.get('Datum', 'NAD 27')).strip()
                        
                        # Texas State Plane zones
                        # NAD 27 uses different EPSG codes than NAD 83
                        if 'NAD 27' in datum or 'NAD27' in datum:
                            zone_epsgs = {
                                'North': 'EPSG:32037',
                                'North Central': 'EPSG:32038', 
                                'Central': 'EPSG:32039',
                                'South Central': 'EPSG:32040',
                                'South': 'EPSG:32041'
                            }
                        else:  # NAD 83
                            zone_epsgs = {
                                'North': 'EPSG:32138',
                                'North Central': 'EPSG:32139',
                                'Central': 'EPSG:32140',
                                'South Central': 'EPSG:32141',
                                'South': 'EPSG:32142'
                            }
                        
                        epsg = zone_epsgs.get(zone, zone_epsgs.get('Central'))
                        
                        # Transform from State Plane to WGS84 (lat/long)
                        transformer = Transformer.from_crs(epsg, "EPSG:4326", always_xy=True)
                        longitude, latitude = transformer.transform(x, y)
                    except Exception as e:
                        print(f"Error converting State Plane coords for {row.get('Lease_Name')}: {e}")
                        pass
                
                permit = {
                    'state': 'Texas',
                    'lease_name': str(row.get('Lease_Name', '')),
                    'api_number': str(row.get('API_Number', '')),
                    'operator': str(row.get('Operator', '')),
                    'well_number': str(row.get('Well_Number', '')),
                    'county': str(row.get('County', '')),
                    'district': str(row.get('District', '')),
                    'wellbore_profile': str(row.get('Wellbore_Profile', '')),
                    'filed_date': str(row.get('Filed_Date', '')),
                    'latitude': latitude,
                    'longitude': longitude,
                }
                # Add distance if calculated
                if 'distance' in row.index:
                    permit['distance_miles'] = round(float(row['distance']), 2)
                permits.append(permit)
        
        # Try to load Oklahoma permits (if needed)
        # TODO: Add Oklahoma permits loading here if you have that data
        
        return {
            'count': len(permits),
            'permits': permits,
            'search_params': {
                'county': county,
                'lat': lat,
                'lng': lng,
                'radius_miles': radius_miles,
                'limit': limit
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching permits: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)