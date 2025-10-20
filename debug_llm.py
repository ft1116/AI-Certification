#!/usr/bin/env python3
"""
Debug LLM invocation
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def debug_llm():
    """Debug LLM invocation"""
    print("🧪 Debugging LLM Invocation")
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
        
        # Test LLM invocation
        print("🤖 Testing LLM invocation...")
        
        try:
            response = chatbot.llm.invoke(messages)
            print(f"✅ LLM response type: {type(response)}")
            print(f"📝 LLM response: {response}")
            
            if hasattr(response, 'content'):
                print(f"📝 Response content: '{response.content}'")
            else:
                print(f"📝 Response string: '{str(response)}'")
                
        except Exception as e:
            print(f"❌ LLM error: {e}")
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_llm()
