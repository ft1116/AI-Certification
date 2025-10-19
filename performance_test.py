#!/usr/bin/env python3
"""
Performance test to compare 100 vs 200 document retrieval
"""

import time
from mineral_insights_langgraph import run_mineral_query

def test_performance():
    test_queries = [
        "What are typical lease terms for oil and gas drilling in Texas?",
        "Show me Pioneer Natural Resources drilling activity in Midland County",
        "What's the market price for mineral rights in Oklahoma?",
        "How do drilling permits work in Texas?",
        "Compare lease offers between Texas and Oklahoma"
    ]
    
    print("🚀 Performance Test - 200 Document Retrieval")
    print("="*60)
    
    total_time = 0
    for i, query in enumerate(test_queries, 1):
        print(f"\n📋 Test {i}/5: {query[:50]}...")
        
        start_time = time.time()
        result = run_mineral_query(query, f"perf_test_{i}")
        end_time = time.time()
        
        query_time = end_time - start_time
        total_time += query_time
        
        print(f"⏱️  Time: {query_time:.2f} seconds")
        print(f"📊 Sources: {len(result.get('sources', []))}")
        print(f"📊 Confidence: {result.get('confidence', 0)}")
    
    avg_time = total_time / len(test_queries)
    print(f"\n📊 Performance Summary:")
    print(f"   Total time: {total_time:.2f} seconds")
    print(f"   Average per query: {avg_time:.2f} seconds")
    print(f"   Expected with 100 docs: ~10-12 seconds")
    print(f"   Performance impact: {avg_time - 11:.2f} seconds slower")

if __name__ == "__main__":
    test_performance()
