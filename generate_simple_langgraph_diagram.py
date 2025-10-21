#!/usr/bin/env python3
"""
Generate Simple LangGraph Workflow Diagram
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# Set up the figure
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
ax.set_xlim(0, 12)
ax.set_ylim(0, 8)
ax.axis('off')

# Define colors
colors = {
    'node': '#E6F3FF',    # Light blue
    'decision': '#FFE6CC', # Light orange
    'web': '#E6FFE6',     # Light green
    'end': '#F0E68C'      # Light yellow
}

# Simple nodes
nodes = [
    {'pos': (6, 7), 'text': 'Query Entry', 'color': colors['node']},
    {'pos': (2, 5.5), 'text': 'Multi-Query\nRetrieval', 'color': colors['node']},
    {'pos': (6, 5.5), 'text': 'Smart\nRanking', 'color': colors['node']},
    {'pos': (10, 5.5), 'text': 'Answer\nGeneration', 'color': colors['node']},
    {'pos': (6, 4), 'text': 'Confidence\nCheck', 'color': colors['decision']},
    {'pos': (2, 2.5), 'text': 'Web Search\n(Tavily)', 'color': colors['web']},
    {'pos': (6, 2.5), 'text': 'Enhanced\nAnswer', 'color': colors['node']},
    {'pos': (10, 2.5), 'text': 'Final\nResponse', 'color': colors['end']}
]

# Draw simple rectangles
for node in nodes:
    x, y = node['pos']
    
    # Simple rectangle
    rect = FancyBboxPatch(
        (x-1, y-0.4), 2, 0.8,
        boxstyle="round,pad=0.1",
        facecolor=node['color'],
        edgecolor='black',
        linewidth=1
    )
    ax.add_patch(rect)
    
    # Text
    ax.text(x, y, node['text'], ha='center', va='center', 
            fontsize=10, fontweight='bold')

# Simple arrows
arrows = [
    ((6, 6.6), (2, 5.9)),   # Entry to Retrieval
    ((6, 6.6), (6, 5.9)),   # Entry to Ranking  
    ((6, 6.6), (10, 5.9)),  # Entry to Generation
    ((3, 5.5), (5, 5.5)),   # Retrieval to Ranking
    ((7, 5.5), (9, 5.5)),   # Ranking to Generation
    ((10, 5.1), (6, 4.4)),  # Generation to Confidence
    ((6, 3.6), (2, 2.9)),   # Confidence to Web (Low)
    ((6, 3.6), (6, 2.9)),   # Confidence to Enhanced (High)
    ((2, 2.1), (6, 2.1)),   # Web to Enhanced
    ((6, 2.1), (10, 2.1))   # Enhanced to Final
]

# Draw arrows
for start, end in arrows:
    arrow = patches.FancyArrowPatch(
        start, end,
        arrowstyle='->',
        mutation_scale=15,
        color='black',
        linewidth=1.5
    )
    ax.add_patch(arrow)

# Add labels for key decisions
ax.text(4, 3.2, 'Low Confidence', ha='center', va='center', 
        fontsize=8, style='italic', color='red')
ax.text(8, 3.2, 'High Confidence', ha='center', va='center', 
        fontsize=8, style='italic', color='green')

# Title
ax.text(6, 7.5, 'LangGraph Orchestrator Workflow', 
        ha='center', va='center', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('langgraph_workflow_diagram.png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ Simple LangGraph workflow diagram generated: langgraph_workflow_diagram.png")
