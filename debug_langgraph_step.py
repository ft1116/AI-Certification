#!/usr/bin/env python3
"""
Debug LangGraph step by step
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def debug_langgraph():
    """Debug LangGraph execution step by step"""
    print("🧪 Debugging LangGraph Step by Step")
    print("=" * 50)
    
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
        print(f"📋 Result keys: {list(result.keys())}")
        
        # Check each key
        for key, value in result.items():
            if key == "streaming_messages":
                print(f"📝 {key}: {len(value) if isinstance(value, list) else type(value)} messages")
                if isinstance(value, list) and value:
                    print(f"   First message: {value[0]}")
            elif key == "final_answer":
                print(f"📝 {key}: '{value}'")
            elif key == "confidence_score":
                print(f"📊 {key}: {value}")
            elif key == "sources_used":
                print(f"📚 {key}: {len(value) if isinstance(value, list) else type(value)} sources")
            else:
                print(f"📝 {key}: {type(value)} - {str(value)[:100]}...")
        
        # Test if we can get streaming messages
        messages = result.get("streaming_messages", [])
        if messages:
            print(f"\n✅ Found {len(messages)} streaming messages")
            print("📝 First message preview:")
            for i, msg in enumerate(messages[:2]):
                print(f"   {i+1}. {msg}")
        else:
            print("\n❌ No streaming messages found!")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_langgraph()
