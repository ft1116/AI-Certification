#!/usr/bin/env python3
"""
Integration script to modify mineral_insights_langgraph.py with the best retriever

This script shows how to replace the current semantic search retriever
with the best performing retriever from the evaluation.
"""

import os
from dotenv import load_dotenv
from advanced_retrievers_evaluation import AdvancedRetrieversEvaluator

def get_best_retriever():
    """Get the best retriever based on evaluation results"""
    
    # You can either:
    # 1. Run the full evaluation and get results
    # 2. Or manually specify the best retriever based on previous results
    
    # For now, let's assume BM25 or Ensemble performed best
    # You can modify this based on your evaluation results
    
    evaluator = AdvancedRetrieversEvaluator()
    
    # Example: If BM25 was the best performer
    best_retriever = evaluator.create_bm25_retriever()
    
    return best_retriever, "BM25"

def modify_langgraph_with_retriever():
    """Show how to modify the LangGraph to use a different retriever"""
    
    print("🔧 How to modify mineral_insights_langgraph.py with the best retriever:")
    print("="*70)
    
    print("""
1. Import the retriever creation function:
   from advanced_retrievers_evaluation import AdvancedRetrieversEvaluator

2. In the create_complete_mineral_graph() function, replace the retrieve_documents node:

   # OLD CODE (lines ~119-144):
   def retrieve_documents(state: MineralQueryState):
       query = state["query"]
       query_embedding = create_embedding(query)
       semantic_results = qdrant_client.query_points(...)
       semantic_docs = qdrant_to_langchain_docs(semantic_results)
       return {"retrieved_documents": semantic_docs}

   # NEW CODE:
   def retrieve_documents(state: MineralQueryState):
       query = state["query"]
       
       # Create the best retriever (replace with your best performer)
       evaluator = AdvancedRetrieversEvaluator()
       retriever = evaluator.create_bm25_retriever()  # or create_ensemble_retriever()
       
       # Retrieve documents using the new retriever
       docs = retriever.get_relevant_documents(query)
       
       return {"retrieved_documents": docs}

3. Update the rank_documents function if needed:
   - BM25 retrievers already return ranked results
   - You might want to skip or simplify the ranking step

4. Test the modified pipeline:
   python -c "from mineral_insights_langgraph import run_mineral_query; print(run_mineral_query('test query'))"
""")

def create_modified_langgraph_example():
    """Create an example of the modified LangGraph with BM25 retriever"""
    
    example_code = '''
# Example modification for mineral_insights_langgraph.py

def retrieve_documents(state: MineralQueryState):
    """Retrieve documents using BM25 retriever (or your best performer)"""
    query = state["query"]
    
    print(f"🔍 Retrieving documents for: '{query}'")
    
    try:
        # Create BM25 retriever (replace with your best performer)
        evaluator = AdvancedRetrieversEvaluator()
        retriever = evaluator.create_bm25_retriever()
        
        # Retrieve documents
        docs = retriever.get_relevant_documents(query)
        
        print(f"📊 Found {len(docs)} documents using BM25")
        print(f"📚 Total documents retrieved: {len(docs)}")
        
        return {"retrieved_documents": docs}
        
    except Exception as e:
        print(f"⚠️ BM25 retrieval error: {e}")
        # Fallback to original semantic search
        query_embedding = create_embedding(query)
        semantic_results = qdrant_client.query_points(
            collection_name=collection_name,
            query=query_embedding,
            limit=200,
            with_payload=True
        )
        semantic_docs = qdrant_to_langchain_docs(qdrant_results=semantic_results)
        return {"retrieved_documents": semantic_docs}
'''
    
    with open("modified_retrieve_documents_example.py", "w") as f:
        f.write(example_code)
    
    print("📝 Created modified_retrieve_documents_example.py with example code")

def main():
    """Main function"""
    print("🔧 Advanced Retriever Integration Guide")
    print("="*50)
    
    # Show integration instructions
    modify_langgraph_with_retriever()
    
    # Create example code
    create_modified_langgraph_example()
    
    print("\n✅ Integration guide complete!")
    print("\nNext steps:")
    print("1. Run: python advanced_retrievers_evaluation.py")
    print("2. Identify the best performing retriever")
    print("3. Modify mineral_insights_langgraph.py with the best retriever")
    print("4. Test the modified pipeline")

if __name__ == "__main__":
    main()

