#!/usr/bin/env python3
"""
Generate Current Mineral Insights System Architecture Diagram
Based on actual implementation: FastAPI + LangGraph + Mapping Agent
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

# Set up the figure
fig, ax = plt.subplots(1, 1, figsize=(14, 10))
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis('off')

# Define colors
colors = {
    'frontend': '#ADD8E6',    # Light blue
    'backend': '#90EE90',     # Light green  
    'langgraph': '#FFD700',   # Gold
    'agent': '#FFB6C1',       # Light pink
    'database': '#87CEEB',    # Sky blue
    'api': '#F0E68C',         # Khaki
    'llm': '#DDA0DD'          # Plum
}

# Define components with positions
components = {
    'frontend': {'pos': (7, 9), 'size': (2.5, 0.8), 'text': 'React Frontend\n(TypeScript + OpenLayers)', 'color': colors['frontend']},
    'fastapi': {'pos': (7, 7.5), 'size': (2.5, 0.8), 'text': 'FastAPI Backend\n(chatbot.py)', 'color': colors['backend']},
    'langgraph': {'pos': (7, 6), 'size': (2.5, 0.8), 'text': 'LangGraph Orchestrator\n(langgraph_chatbot.py)', 'color': colors['langgraph']},
    'mapping_agent': {'pos': (11, 5), 'size': (2, 0.8), 'text': 'Mapping Agent\n(mapping_agent.py)', 'color': colors['agent']},
    'pinecone': {'pos': (3, 4.5), 'size': (2, 0.8), 'text': 'Pinecone Vector DB\n(27K documents)', 'color': colors['database']},
    'sqlite': {'pos': (11, 3.5), 'size': (2, 0.8), 'text': 'SQLite Permits DB\n(5.6K permits)', 'color': colors['database']},
    'tavily': {'pos': (3, 3), 'size': (2, 0.8), 'text': 'Tavily Web Search\n(Real-time)', 'color': colors['api']},
    'claude': {'pos': (7, 2), 'size': (2, 0.8), 'text': 'Claude Sonnet 4.5\n(LLM)', 'color': colors['llm']},
    'openai_emb': {'pos': (11, 2), 'size': (2, 0.8), 'text': 'OpenAI Embeddings\n(text-embedding-3-small)', 'color': colors['llm']}
}

# Draw components
for comp_id, comp in components.items():
    x, y = comp['pos']
    width, height = comp['size']
    
    # Create rounded rectangle
    bbox = FancyBboxPatch(
        (x - width/2, y - height/2), width, height,
        boxstyle="round,pad=0.1",
        facecolor=comp['color'],
        edgecolor='black',
        linewidth=1.5
    )
    ax.add_patch(bbox)
    
    # Add text
    ax.text(x, y, comp['text'], ha='center', va='center', 
            fontsize=9, fontweight='bold')

# Define connections
connections = [
    # Main flow
    ('frontend', 'fastapi', 'HTTP/SSE'),
    ('fastapi', 'langgraph', 'Query Processing'),
    
    # LangGraph connections
    ('langgraph', 'pinecone', 'Document Retrieval'),
    ('langgraph', 'tavily', 'Web Search (Low Confidence)'),
    ('langgraph', 'claude', 'Answer Generation'),
    
    # FastAPI to Mapping Agent
    ('fastapi', 'mapping_agent', 'Mapping Query?'),
    ('mapping_agent', 'sqlite', 'Permit Queries'),
    
    # External connections
    ('pinecone', 'openai_emb', 'Embeddings'),
    ('tavily', 'claude', 'Web Results'),
    ('sqlite', 'fastapi', 'GeoJSON Data'),
    ('claude', 'langgraph', 'Generated Answers'),
    ('langgraph', 'fastapi', 'Final Response'),
    ('fastapi', 'frontend', 'Streaming Response')
]

# Draw connections
for start, end, label in connections:
    start_pos = components[start]['pos']
    end_pos = components[end]['pos']
    
    # Calculate arrow direction
    dx = end_pos[0] - start_pos[0]
    dy = end_pos[1] - start_pos[1]
    
    # Draw arrow
    arrow = patches.FancyArrowPatch(
        start_pos, end_pos,
        arrowstyle='->',
        mutation_scale=15,
        color='black',
        linewidth=1.5
    )
    ax.add_patch(arrow)
    
    # Add label for key connections
    if label in ['HTTP/SSE', 'Query Processing', 'Mapping Query?', 'Streaming Response']:
        mid_x = (start_pos[0] + end_pos[0]) / 2
        mid_y = (start_pos[1] + end_pos[1]) / 2
        
        # Offset label position
        if label == 'Mapping Query?':
            mid_x += 0.3
            mid_y += 0.2
        elif label == 'Streaming Response':
            mid_x -= 0.3
            mid_y += 0.2
        
        ax.text(mid_x, mid_y, label, ha='center', va='center', 
                fontsize=8, bbox=dict(boxstyle="round,pad=0.2", 
                facecolor='white', alpha=0.8))

# Add title
ax.text(7, 9.7, 'Mineral Insights: Current System Architecture', 
        ha='center', va='center', fontsize=16, fontweight='bold')

# Add legend
legend_elements = [
    patches.Patch(color=colors['frontend'], label='Frontend'),
    patches.Patch(color=colors['backend'], label='Backend'),
    patches.Patch(color=colors['langgraph'], label='LangGraph'),
    patches.Patch(color=colors['agent'], label='AI Agent'),
    patches.Patch(color=colors['database'], label='Database'),
    patches.Patch(color=colors['api'], label='External API'),
    patches.Patch(color=colors['llm'], label='LLM/Embeddings')
]

ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1))

# Add key features
features_text = """Key Features:
• Multi-Query Decomposition
• Smart Document Ranking
• Confidence-Based Web Search
• Streaming Responses (SSE)
• Interactive Mapping
• Location Intelligence"""

ax.text(0.5, 0.3, features_text, transform=ax.transAxes, 
        fontsize=10, verticalalignment='bottom',
        bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.7))

plt.tight_layout()
plt.savefig('mineral_insights_current_architecture.png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ Current architecture diagram generated: mineral_insights_current_architecture.png")
