#!/usr/bin/env python3
"""
Analyze why Context Precision is low, especially for geological and procedural queries
"""

import os
import pandas as pd
from mineral_insights_langgraph import run_mineral_query
from dotenv import load_dotenv

load_dotenv()

def analyze_query_precision(question, expected_context_types=None):
    """Analyze a specific query to understand precision issues"""
    
    print(f"\n🔍 ANALYZING QUERY: '{question}'")
    print("="*80)
    
    # Run the query
    result = run_mineral_query(question, f"precision_analysis")
    
    # Extract results
    answer = result.get("answer", "")
    sources = result.get("sources", [])
    confidence = result.get("confidence", 0.0)
    
    print(f"📊 Confidence: {confidence}")
    print(f"📊 Sources found: {len(sources)}")
    print(f"📊 Sources: {sources}")
    
    # Analyze answer quality
    print(f"\n📝 Answer Preview:")
    print(f"{answer[:300]}...")
    
    # Analyze source relevance
    print(f"\n🔍 Source Analysis:")
    for i, source in enumerate(sources, 1):
        print(f"  {i}. {source}")
        
        # Check if source contains relevant keywords
        source_lower = source.lower()
        question_lower = question.lower()
        
        # Extract key terms from question
        key_terms = []
        if "formation" in question_lower:
            key_terms.extend(["formation", "geological", "target", "drill"])
        if "permit" in question_lower:
            key_terms.extend(["permit", "drilling", "approval", "process"])
        if "royalty" in question_lower:
            key_terms.extend(["royalty", "rate", "percentage", "revenue"])
        if "environmental" in question_lower:
            key_terms.extend(["environmental", "impact", "regulation", "safety"])
        
        # Check relevance
        relevant_terms_found = [term for term in key_terms if term in source_lower]
        print(f"     Relevant terms found: {relevant_terms_found}")
        print(f"     Relevance: {'HIGH' if len(relevant_terms_found) >= 2 else 'MEDIUM' if len(relevant_terms_found) >= 1 else 'LOW'}")
    
    return {
        'question': question,
        'sources': sources,
        'answer': answer,
        'confidence': confidence
    }

def test_problematic_queries():
    """Test the queries that had 0.000 precision"""
    
    print("🧪 TESTING QUERIES WITH 0.000 CONTEXT PRECISION")
    print("="*80)
    
    # These are the queries that had 0.000 precision
    problematic_queries = [
        {
            'question': 'How do drilling permits work in Texas?',
            'type': 'procedural',
            'expected': ['permit', 'process', 'approval', 'drilling', 'texas']
        },
        {
            'question': 'What formations are being targeted in Oklahoma drilling permits?',
            'type': 'geological', 
            'expected': ['formation', 'oklahoma', 'target', 'geological', 'drill']
        }
    ]
    
    results = []
    
    for query_data in problematic_queries:
        result = analyze_query_precision(
            query_data['question'], 
            query_data['expected']
        )
        results.append(result)
    
    return results

def analyze_data_coverage():
    """Analyze what types of data we have in the database"""
    
    print("\n📊 ANALYZING DATA COVERAGE")
    print("="*80)
    
    # Test different query types to see what data we have
    test_queries = [
        "Texas drilling permits",
        "Oklahoma formations", 
        "lease terms and conditions",
        "royalty rates",
        "environmental regulations",
        "drilling processes",
        "geological formations",
        "permit applications"
    ]
    
    coverage_results = {}
    
    for query in test_queries:
        print(f"\n🔍 Testing: '{query}'")
        result = run_mineral_query(query, f"coverage_test")
        sources = result.get("sources", [])
        
        # Categorize sources
        source_types = {
            'lease_offers': 0,
            'permits': 0, 
            'forum': 0,
            'mineral_offers': 0,
            'other': 0
        }
        
        for source in sources:
            source_lower = source.lower()
            if 'lease' in source_lower:
                source_types['lease_offers'] += 1
            elif 'permit' in source_lower:
                source_types['permits'] += 1
            elif 'forum' in source_lower:
                source_types['forum'] += 1
            elif 'mineral' in source_lower:
                source_types['mineral_offers'] += 1
            else:
                source_types['other'] += 1
        
        coverage_results[query] = {
            'total_sources': len(sources),
            'source_types': source_types
        }
        
        print(f"  Sources: {len(sources)}")
        print(f"  Types: {source_types}")
    
    return coverage_results

def main():
    """Main analysis function"""
    
    print("🔬 CONTEXT PRECISION ANALYSIS")
    print("="*80)
    print("Analyzing why geological and procedural queries have 0.000 precision...")
    
    # Test problematic queries
    problematic_results = test_problematic_queries()
    
    # Analyze data coverage
    coverage_results = analyze_data_coverage()
    
    # Summary
    print("\n📋 ANALYSIS SUMMARY")
    print("="*80)
    
    print("\n🔴 Issues Found:")
    print("1. Geological queries may not have relevant formation data in sources")
    print("2. Procedural queries may not have process/regulatory information")
    print("3. Sources may be too generic (e.g., 'Lease Offer' without specific details)")
    print("4. Ranking algorithm may not be matching query intent to document content")
    
    print("\n💡 Recommendations:")
    print("1. Improve source descriptions to be more specific")
    print("2. Add more geological and procedural data to the database")
    print("3. Enhance ranking algorithm for better semantic matching")
    print("4. Consider query expansion for technical terms")
    
    return {
        'problematic_results': problematic_results,
        'coverage_results': coverage_results
    }

if __name__ == "__main__":
    main()
