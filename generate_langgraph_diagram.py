#!/usr/bin/env python3
"""
Generate LangGraph Workflow Diagram for Mineral Insights
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
    'entry': '#E6F3FF',      # Light blue for entry point
    'retrieval': '#FFE6CC',  # Light orange for retrieval
    'ranking': '#E6FFE6',    # Light green for ranking
    'generation': '#FFE6F3', # Light pink for generation
    'decision': '#F0E68C',   # Khaki for decision points
    'web_search': '#E6E6FA', # Lavender for web search
    'validation': '#F5F5DC', # Beige for validation
    'end': '#D3D3D3'         # Light gray for end
}

# Define nodes with positions and properties
nodes = {
    'entry': {'pos': (7, 9), 'size': (1.5, 0.6), 'text': 'Query\nEntry', 'color': colors['entry']},
    'retrieve': {'pos': (3, 7.5), 'size': (2, 0.8), 'text': 'retrieve_documents\n(Multi-Query + Pinecone)', 'color': colors['retrieval']},
    'rank': {'pos': (7, 7.5), 'size': (2, 0.8), 'text': 'rank_documents\n(Smart Ranking)', 'color': colors['ranking']},
    'generate': {'pos': (11, 7.5), 'size': (2, 0.8), 'text': 'generate_answer\n(Confidence Scoring)', 'color': colors['generation']},
    'decide': {'pos': (7, 5.5), 'size': (2, 0.6), 'text': 'should_search_web\n(Confidence Check)', 'color': colors['decision']},
    'web_search': {'pos': (3, 4), 'size': (2, 0.8), 'text': 'tavily_web_search\n(Real-time Web)', 'color': colors['web_search']},
    'enhanced': {'pos': (7, 4), 'size': (2, 0.8), 'text': 'generate_enhanced_answer\n(RAG + Web Data)', 'color': colors['generation']},
    'validate': {'pos': (11, 4), 'size': (2, 0.8), 'text': 'validate_answer\n(Quality Check)', 'color': colors['validation']},
    'mapping': {'pos': (7, 2.5), 'size': (2, 0.6), 'text': 'check_for_mapping_query\n(Pass-through)', 'color': colors['decision']},
    'end': {'pos': (7, 1), 'size': (1.5, 0.6), 'text': 'Final\nResponse', 'color': colors['end']}
}

# Draw nodes
for node_id, node in nodes.items():
    x, y = node['pos']
    width, height = node['size']
    
    # Create rounded rectangle
    bbox = FancyBboxPatch(
        (x - width/2, y - height/2), width, height,
        boxstyle="round,pad=0.1",
        facecolor=node['color'],
        edgecolor='black',
        linewidth=1.5
    )
    ax.add_patch(bbox)
    
    # Add text
    ax.text(x, y, node['text'], ha='center', va='center', 
            fontsize=9, fontweight='bold', wrap=True)

# Define arrows with labels
arrows = [
    ('entry', 'retrieve', ''),
    ('entry', 'rank', ''),
    ('entry', 'generate', ''),
    ('retrieve', 'rank', 'Documents'),
    ('rank', 'generate', 'Ranked Docs'),
    ('generate', 'decide', 'Answer + Confidence'),
    ('decide', 'web_search', 'Low Confidence'),
    ('decide', 'validate', 'High Confidence'),
    ('web_search', 'enhanced', 'Web Results'),
    ('enhanced', 'validate', 'Enhanced Answer'),
    ('validate', 'mapping', 'Validated Answer'),
    ('mapping', 'end', 'Final Response')
]

# Draw arrows
for start, end, label in arrows:
    start_pos = nodes[start]['pos']
    end_pos = nodes[end]['pos']
    
    # Calculate arrow direction and position
    dx = end_pos[0] - start_pos[0]
    dy = end_pos[1] - start_pos[1]
    
    # Draw arrow
    arrow = patches.FancyArrowPatch(
        start_pos, end_pos,
        arrowstyle='->',
        mutation_scale=20,
        color='black',
        linewidth=2
    )
    ax.add_patch(arrow)
    
    # Add label if present
    if label:
        mid_x = (start_pos[0] + end_pos[0]) / 2
        mid_y = (start_pos[1] + end_pos[1]) / 2
        
        # Offset label position
        if 'Low Confidence' in label:
            mid_x -= 0.5
            mid_y -= 0.3
        elif 'High Confidence' in label:
            mid_x += 0.5
            mid_y -= 0.3
        
        ax.text(mid_x, mid_y, label, ha='center', va='center', 
                fontsize=8, bbox=dict(boxstyle="round,pad=0.2", 
                facecolor='white', alpha=0.8))

# Add title
ax.text(7, 9.7, 'LangGraph Orchestrator Workflow', 
        ha='center', va='center', fontsize=16, fontweight='bold')

# Add legend
legend_elements = [
    patches.Patch(color=colors['retrieval'], label='Document Retrieval'),
    patches.Patch(color=colors['ranking'], label='Document Ranking'),
    patches.Patch(color=colors['generation'], label='Answer Generation'),
    patches.Patch(color=colors['decision'], label='Decision Points'),
    patches.Patch(color=colors['web_search'], label='Web Search'),
    patches.Patch(color=colors['validation'], label='Validation')
]

ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1))

# Add key features box
features_text = """Key Features:
• Multi-Query Decomposition
• Smart Document Ranking (7+ factors)
• Confidence-Based Web Search
• Streaming Response Generation
• Conditional Formatting"""

ax.text(0.5, 0.5, features_text, transform=ax.transAxes, 
        fontsize=10, verticalalignment='bottom',
        bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.7))

plt.tight_layout()
plt.savefig('langgraph_workflow_diagram.png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ LangGraph workflow diagram generated: langgraph_workflow_diagram.png")
