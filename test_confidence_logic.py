#!/usr/bin/env python3
"""
Test the new confidence logic and Tavily triggering
"""

from mineral_insights_langgraph import run_mineral_query

def test_confidence_queries():
    """Test different query types to see confidence scores and Tavily triggering"""
    
    test_queries = [
        # Should trigger Tavily (current/recent information)
        "What are the current trends in mineral rights transactions?",
        "What's the latest market price for oil and gas?",
        "Show me recent drilling activity in the Permian Basin",
        
        # Should trigger Tavily (comparison queries)
        "Compare lease offers between Texas and Oklahoma",
        
        # Should NOT trigger Tavily (specific factual queries)
        "What formations are being targeted in Oklahoma drilling permits?",
        "Show me Pioneer Natural Resources drilling activity in Midland County",
        "How do drilling permits work in Texas?",
        
        # Borderline cases
        "What are typical lease terms for oil and gas drilling in Texas?",
        "What's the market price for mineral rights in Oklahoma?"
    ]
    
    print("🧪 Testing New Confidence Logic and Tavily Triggering")
    print("="*80)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n📋 Test {i}: {query}")
        print("-" * 60)
        
        try:
            result = run_mineral_query(query, f"confidence_test_{i}")
            confidence = result.get('confidence', 0.0)
            sources = result.get('sources', [])
            
            # Determine if Tavily would trigger
            tavily_triggered = confidence < 0.8
            
            print(f"📊 Confidence: {confidence:.3f}")
            print(f"🌐 Tavily Triggered: {'YES' if tavily_triggered else 'NO'}")
            print(f"📚 Sources: {len(sources)}")
            print(f"💬 Answer Preview: {result.get('answer', '')[:150]}...")
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n" + "="*80)
    print("✅ Confidence logic testing complete!")

if __name__ == "__main__":
    test_confidence_queries()

