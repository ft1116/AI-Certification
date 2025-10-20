#!/usr/bin/env python3
"""
Debug streaming specifically
"""

import os
from dotenv import load_dotenv
import asyncio

# Load environment variables
load_dotenv()

async def debug_streaming():
    """Debug streaming specifically"""
    print("🧪 Debugging Streaming")
    print("=" * 30)
    
    try:
        from langgraph_chatbot import get_chatbot
        
        print("✅ Successfully imported langgraph_chatbot")
        
        # Get chatbot instance
        chatbot = get_chatbot()
        print("✅ Successfully created chatbot instance")
        
        # Test query
        query = "What are typical lease terms in Texas?"
        print(f"🔍 Testing query: {query}")
        
        # Create initial state
        initial_state = {
            "query": query,
            "conversation_history": [],
            "needs_web_search": False,
            "web_search_results": ""
        }
        
        print("📊 Running LangGraph...")
        
        # Run the graph
        result = chatbot.graph.invoke(initial_state)
        
        print("✅ LangGraph completed!")
        
        # Get the streaming messages
        messages = result.get("streaming_messages", [])
        
        if not messages:
            print("❌ No streaming messages found!")
            return
        
        print(f"✅ Found {len(messages)} streaming messages")
        
        # Test streaming directly
        print("📡 Testing LLM streaming...")
        
        try:
            full_response = ""
            chunk_count = 0
            async for chunk in chatbot.llm.astream(messages):
                chunk_count += 1
                if hasattr(chunk, 'content') and chunk.content:
                    content = chunk.content
                    full_response += content
                    print(f"Chunk {chunk_count}: '{content}'")
                elif isinstance(chunk, str):
                    full_response += chunk
                    print(f"Chunk {chunk_count}: '{chunk}'")
                
                # Limit to first few chunks for testing
                if chunk_count >= 5:
                    break
            
            print(f"\n✅ Streaming successful! Total chunks: {chunk_count}")
            print(f"📝 Full response so far: '{full_response}'")
            
        except Exception as e:
            print(f"❌ Streaming error: {e}")
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_streaming())
