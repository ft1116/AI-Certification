#!/usr/bin/env python3
"""
Generate a PNG diagram of the Mineral Insights workflow
"""

import graphviz

def generate_workflow_diagram(filename="mineral_insights_workflow.png"):
    dot = graphviz.Digraph(comment='Mineral Insights Workflow', graph_attr={'rankdir': 'TB', 'splines': 'curved'})
    
    # User and Frontend
    dot.node('A', 'User Query', shape='ellipse', style='filled', fillcolor='#E6F3FF')
    dot.node('R', 'React Frontend', shape='box', style='filled', fillcolor='#ADD8E6')
    
    # FastAPI Backend
    dot.node('B', 'FastAPI Backend', shape='box', style='filled', fillcolor='#90EE90')
    
    # Location and Mapping
    dot.node('C', 'Location Extraction', shape='box', style='filled', fillcolor='#FFB6C1')
    dot.node('N', 'Mapping Query?', shape='diamond', style='filled', fillcolor='#FFD700')
    dot.node('O', 'Mapping Agent', shape='box', style='filled', fillcolor='#FFB6C1')
    dot.node('P', 'SQLite Permit DB', shape='cylinder', style='filled', fillcolor='#87CEEB')
    
    # LangGraph Orchestrator
    dot.node('D', 'LangGraph Orchestrator', shape='box', style='filled', fillcolor='#FFD700')
    dot.node('E', 'Document Retrieval', shape='box', style='filled', fillcolor='#DDA0DD')
    dot.node('F', 'Pinecone Vector DB', shape='cylinder', style='filled', fillcolor='#87CEEB')
    dot.node('G', 'Document Ranking', shape='box', style='filled', fillcolor='#DDA0DD')
    dot.node('H', 'Answer Generation', shape='box', style='filled', fillcolor='#DDA0DD')
    
    # Confidence and Web Search
    dot.node('I', 'Confidence Check', shape='diamond', style='filled', fillcolor='#FFD700')
    dot.node('J', 'Web Search Agent', shape='box', style='filled', fillcolor='#FFB6C1')
    dot.node('K', 'Tavily API', shape='oval', style='filled', fillcolor='#F0E68C')
    dot.node('L', 'Enhanced Answer Generation', shape='box', style='filled', fillcolor='#DDA0DD')
    
    # Final Response
    dot.node('M', 'Final Answer', shape='box', style='filled', fillcolor='#98FB98')
    dot.node('Q', 'Response Assembly', shape='box', style='filled', fillcolor='#90EE90')
    
    # Main flow
    dot.edge('A', 'B', label='API Request')
    dot.edge('B', 'C', label='Extract Location')
    dot.edge('B', 'D', label='Process Query')
    dot.edge('D', 'E', label='Retrieve Documents')
    dot.edge('E', 'F', label='Query Vector DB')
    dot.edge('D', 'G', label='Rank Documents')
    dot.edge('D', 'H', label='Generate Answer')
    dot.edge('H', 'I', label='Check Confidence')
    dot.edge('I', 'J', label='Low Confidence')
    dot.edge('J', 'K', label='Web Search')
    dot.edge('K', 'L', label='Enhance Answer')
    dot.edge('I', 'M', label='High Confidence')
    dot.edge('L', 'M', label='Final Answer')
    
    # Mapping flow
    dot.edge('B', 'N', label='Check for Mapping')
    dot.edge('N', 'O', label='Yes')
    dot.edge('O', 'P', label='Query Permits')
    
    # Response assembly
    dot.edge('M', 'Q', label='Assemble Response')
    dot.edge('P', 'Q', label='Add Mapping Data')
    dot.edge('Q', 'R', label='Return to Frontend')
    
    # Add styling
    dot.attr('node', fontsize='10')
    dot.attr('edge', fontsize='8')
    
    # Render the diagram
    dot.render(filename.replace('.png', ''), format='png', cleanup=True)
    print(f"✅ Workflow diagram saved as: {filename}")
    print("📊 Diagram shows:")
    print("   • User Query → FastAPI Backend")
    print("   • Location Extraction & Mapping Detection")
    print("   • LangGraph Orchestrator with Document Processing")
    print("   • Confidence-based Web Search")
    print("   • Response Assembly with Mapping Data")
    print("   • Return to React Frontend")

if __name__ == "__main__":
    generate_workflow_diagram()
