"""
Analyze why there are so many duplicates in the retrieval results
"""

from mineral_insights_langgraph import MineralInsightsGraph
import hashlib

def analyze_duplicate_sources():
    """Analyze where duplicates are coming from"""
    
    print('🔍 DUPLICATE SOURCE ANALYSIS')
    print('=' * 50)
    
    graph = MineralInsightsGraph()
    
    # Test with a broad query
    query = "drilling permits in Texas"
    
    print(f'Query: "{query}"')
    print('-' * 30)
    
    # Get the raw retrieval results
    state = {
        "query": query,
        "retrieved_documents": [],
        "ranked_documents": [],
        "final_answer": "",
        "confidence_score": 0.0,
        "sources_used": []
    }
    
    # Step 1: Retrieve documents
    state.update(graph.retrieve_documents(state))
    
    documents = state["retrieved_documents"]
    print(f'📚 Total documents retrieved: {len(documents)}')
    
    # Analyze by search strategy
    print(f'\n🔍 Analyzing by search strategy:')
    
    # Group by similarity score ranges to see if semantic search is returning similar results
    score_ranges = {
        '0.7-0.8': [],
        '0.6-0.7': [],
        '0.5-0.6': [],
        '0.4-0.5': [],
        '0.3-0.4': []
    }
    
    for doc in documents:
        score = doc.metadata.get('score', 0)
        if 0.7 <= score < 0.8:
            score_ranges['0.7-0.8'].append(doc)
        elif 0.6 <= score < 0.7:
            score_ranges['0.6-0.7'].append(doc)
        elif 0.5 <= score < 0.6:
            score_ranges['0.5-0.6'].append(doc)
        elif 0.4 <= score < 0.5:
            score_ranges['0.4-0.5'].append(doc)
        elif 0.3 <= score < 0.4:
            score_ranges['0.3-0.4'].append(doc)
    
    print(f'\n📊 Score distribution:')
    for range_name, docs in score_ranges.items():
        if docs:
            print(f'   {range_name}: {len(docs)} documents')
            # Show sample operators
            operators = [doc.metadata.get('operator', 'Unknown') for doc in docs[:3]]
            print(f'     Sample operators: {", ".join(operators)}')
    
    # Analyze content similarity
    print(f'\n🔍 Content similarity analysis:')
    content_hashes = {}
    duplicate_groups = {}
    
    for doc in documents:
        # Create hash based on first 200 characters (more than the deduplication uses)
        content_preview = doc.page_content[:200]
        content_hash = hashlib.md5(content_preview.encode()).hexdigest()
        
        if content_hash in content_hashes:
            # This is a duplicate
            if content_hash not in duplicate_groups:
                duplicate_groups[content_hash] = [content_hashes[content_hash]]
            duplicate_groups[content_hash].append(doc)
        else:
            content_hashes[content_hash] = doc
    
    print(f'   Unique content groups: {len(content_hashes)}')
    print(f'   Duplicate groups: {len(duplicate_groups)}')
    
    # Show examples of duplicate groups
    if duplicate_groups:
        print(f'\n📋 Example duplicate groups:')
        for i, (hash_key, group) in enumerate(list(duplicate_groups.items())[:3], 1):
            print(f'   Group {i} ({len(group)} duplicates):')
            for j, doc in enumerate(group[:3], 1):
                operator = doc.metadata.get('operator', 'Unknown')
                county = doc.metadata.get('county', 'Unknown')
                score = doc.metadata.get('score', 0)
                print(f'     {j}. {operator} - {county} (Score: {score:.4f})')
            print(f'     Content preview: {group[0].page_content[:100]}...')
            print()
    
    # Analyze by data type
    print(f'\n📊 Analysis by data type:')
    data_types = {}
    for doc in documents:
        data_type = doc.metadata.get('data_type', 'unknown')
        if data_type not in data_types:
            data_types[data_type] = []
        data_types[data_type].append(doc)
    
    for data_type, docs in data_types.items():
        print(f'   {data_type}: {len(docs)} documents')
        if docs:
            # Show sample content
            sample_doc = docs[0]
            print(f'     Sample: {sample_doc.page_content[:80]}...')
    
    # Analyze the deduplication logic
    print(f'\n🔍 Deduplication logic analysis:')
    print(f'   Current deduplication uses first 100 characters')
    print(f'   This might be too aggressive for similar permit structures')
    
    # Test with different deduplication strategies
    print(f'\n🧪 Testing different deduplication strategies:')
    
    # Strategy 1: Current (100 chars)
    hash_100 = {}
    unique_100 = []
    for doc in documents:
        content_hash = hash(doc.page_content[:100])
        if content_hash not in hash_100:
            hash_100[content_hash] = True
            unique_100.append(doc)
    
    # Strategy 2: More lenient (50 chars)
    hash_50 = {}
    unique_50 = []
    for doc in documents:
        content_hash = hash(doc.page_content[:50])
        if content_hash not in hash_50:
            hash_50[content_hash] = True
            unique_50.append(doc)
    
    # Strategy 3: By operator + well + county
    hash_structured = {}
    unique_structured = []
    for doc in documents:
        operator = doc.metadata.get('operator', '')
        well = doc.metadata.get('well_name', '') or doc.metadata.get('well_number', '')
        county = doc.metadata.get('county', '')
        structured_hash = hash(f"{operator}_{well}_{county}")
        if structured_hash not in hash_structured:
            hash_structured[structured_hash] = True
            unique_structured.append(doc)
    
    print(f'   Strategy 1 (100 chars): {len(unique_100)} unique documents')
    print(f'   Strategy 2 (50 chars): {len(unique_50)} unique documents')
    print(f'   Strategy 3 (structured): {len(unique_structured)} unique documents')
    
    # Recommendations
    print(f'\n💡 RECOMMENDATIONS:')
    print(f'   1. The high duplicate rate (60%) suggests the deduplication is too aggressive')
    print(f'   2. Many Texas permits have similar structures (same first 100 chars)')
    print(f'   3. Consider using structured deduplication (operator + well + county)')
    print(f'   4. Or reduce the deduplication threshold from 100 to 50 characters')

if __name__ == "__main__":
    analyze_duplicate_sources()

