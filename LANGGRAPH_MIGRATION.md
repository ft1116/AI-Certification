# LangGraph Integration Migration Guide

## Overview
This guide documents the migration from LangChain + Pinecone to LangGraph + Qdrant + Tavily for the Mineral Insights chatbot.

## What Changed

### 1. **New Files Created**
- `langgraph_chatbot.py` - Core LangGraph implementation with streaming support
- `test_langgraph_integration.py` - Test script for the new integration
- `requirements_langgraph.txt` - New dependencies for LangGraph setup

### 2. **Modified Files**
- `chatbot.py` - Updated chat endpoints to use LangGraph instead of LangChain

### 3. **Key Changes in chatbot.py**

#### Chat Endpoints Updated:
- `/chat/stream` - Now uses `stream_mineral_query()` from LangGraph
- `/chat` - Now uses `run_mineral_query()` from LangGraph

#### Preserved Features:
- ✅ Conversation memory (SimpleChatMessageHistory)
- ✅ Location extraction and geocoding
- ✅ PDF upload functionality
- ✅ All existing API endpoints
- ✅ CORS middleware
- ✅ Error handling
- ✅ Streaming responses (Server-Sent Events)

## New Dependencies

### Install LangGraph Dependencies:
```bash
pip install -r requirements_langgraph.txt
```

### Key New Dependencies:
- `langgraph>=0.2.0` - Core LangGraph framework
- `qdrant-client>=1.7.0` - Vector database (replaces Pinecone)
- `tavily-python>=0.3.0` - Web search integration
- `langchain-anthropic>=0.2.0` - Anthropic Claude integration

## Environment Variables

### Required Environment Variables:
```bash
# Qdrant (replaces Pinecone)
QDRANT_URL=http://localhost:6333

# Tavily (new for web search)
TAVILY_API_KEY=your_tavily_api_key

# Anthropic (for Claude)
ANTHROPIC_API_KEY=your_anthropic_api_key

# OpenAI (for embeddings)
OPENAI_API_KEY=your_openai_api_key

# Optional: Keep existing
PINECONE_API_KEY=your_pinecone_api_key  # If still using Pinecone
COHERE_API_KEY=your_cohere_api_key      # If still using Cohere
```

## Data Migration: Pinecone → Qdrant

### 1. **Start Qdrant Server**
```bash
# Using Docker
docker run -p 6333:6333 qdrant/qdrant

# Or install locally
pip install qdrant-client
```

### 2. **Migrate Data**
The existing `ingest_all_data_to_qdrant.py` script should handle the data migration from your current vector store to Qdrant.

### 3. **Verify Data**
```bash
python test_langgraph_integration.py
```

## API Compatibility

### ✅ **Backward Compatible Endpoints:**
- `POST /chat` - Returns same response format + confidence & sources
- `POST /chat/stream` - Same streaming format
- `GET /conversation/{conversation_id}` - Unchanged
- `DELETE /conversation/{conversation_id}` - Unchanged
- `POST /upload-pdf` - Unchanged
- All mapping and permits endpoints - Unchanged

### 🔄 **Enhanced Response Format:**
```json
{
  "answer": "Response text",
  "location": {...},
  "conversation_id": "default",
  "confidence": 0.85,  // NEW: Confidence score
  "sources": [...]      // NEW: Source information
}
```

## Testing the Integration

### 1. **Test LangGraph Directly:**
```bash
python test_langgraph_integration.py
```

### 2. **Test FastAPI Endpoints:**
```bash
# Start the server
python chatbot.py

# Test streaming endpoint
curl -X POST "http://localhost:8003/chat/stream" \
  -H "Content-Type: application/json" \
  -d '{"query": "What are lease rates in Oklahoma?", "conversation_id": "test"}'

# Test regular endpoint
curl -X POST "http://localhost:8003/chat" \
  -H "Content-Type: application/json" \
  -d '{"query": "What are lease rates in Oklahoma?", "conversation_id": "test"}'
```

## Performance Improvements

### 🚀 **LangGraph Benefits:**
1. **Better Context Management** - More sophisticated document ranking
2. **Web Search Integration** - Automatic web search for current information
3. **Confidence Scoring** - Smart confidence calculation for better responses
4. **Streaming Support** - True streaming with LangGraph nodes
5. **Conversation Memory** - Enhanced memory integration

### 📊 **Expected Performance:**
- **Response Quality**: Improved with better context and web search
- **Streaming**: Smoother with LangGraph's async support
- **Memory**: Better conversation context retention
- **Confidence**: More accurate confidence scoring

## Troubleshooting

### Common Issues:

1. **Qdrant Connection Error**
   ```bash
   # Start Qdrant server
   docker run -p 6333:6333 qdrant/qdrant
   ```

2. **Missing Dependencies**
   ```bash
   pip install -r requirements_langgraph.txt
   ```

3. **Environment Variables**
   ```bash
   # Check your .env file has all required variables
   cat .env
   ```

4. **Import Errors**
   ```bash
   # Make sure you're in the right directory
   cd "/Users/fmt116/Desktop/AI Certification"
   python -c "import langgraph_chatbot"
   ```

## Rollback Plan

If you need to rollback to the original LangChain implementation:

1. **Restore Original chatbot.py** (from git or backup)
2. **Remove LangGraph Dependencies** (optional)
3. **Restart with Pinecone** (if you kept the original setup)

## Next Steps

1. **Test the Integration** - Run the test script
2. **Start Qdrant Server** - Ensure vector database is running
3. **Update Environment Variables** - Add Tavily and Qdrant configs
4. **Test Frontend** - Verify the React frontend still works
5. **Monitor Performance** - Check response quality and speed

## Support

For issues with the LangGraph integration:
1. Check the test script output
2. Verify all dependencies are installed
3. Ensure Qdrant server is running
4. Check environment variables are set correctly
