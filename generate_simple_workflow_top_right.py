#!/usr/bin/env python3
"""
Generate Simple Clean Workflow Diagram with Info Box at Top Right
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Set up the figure
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
ax.set_xlim(0, 12)
ax.set_ylim(0, 8)
ax.axis('off')

# Simple color scheme
colors = {
    'blue': '#4A90E2',
    'green': '#7ED321', 
    'orange': '#F5A623',
    'purple': '#9013FE',
    'red': '#D0021B',
    'gray': '#9B9B9B'
}

# Simple components - just rectangles
components = [
    {'pos': (6, 7), 'size': (2, 0.6), 'text': 'User Query', 'color': colors['blue']},
    {'pos': (6, 6), 'size': (2.5, 0.6), 'text': 'FastAPI Backend', 'color': colors['green']},
    {'pos': (6, 5), 'size': (2.5, 0.6), 'text': 'LangGraph Orchestrator', 'color': colors['orange']},
    {'pos': (2, 3.5), 'size': (2, 0.6), 'text': 'Pinecone Vector DB', 'color': colors['purple']},
    {'pos': (10, 3.5), 'size': (2, 0.6), 'text': 'Web Search (Tavily)', 'color': colors['red']},
    {'pos': (6, 2), 'size': (2.5, 0.6), 'text': 'Response Assembly', 'color': colors['green']},
    {'pos': (6, 1), 'size': (2, 0.6), 'text': 'React Frontend', 'color': colors['blue']}
]

# Draw simple rectangles
for comp in components:
    x, y = comp['pos']
    width, height = comp['size']
    
    rect = FancyBboxPatch(
        (x - width/2, y - height/2), width, height,
        boxstyle="round,pad=0.05",
        facecolor=comp['color'],
        edgecolor='white',
        linewidth=2
    )
    ax.add_patch(rect)
    
    # White text
    ax.text(x, y, comp['text'], ha='center', va='center', 
            fontsize=10, fontweight='bold', color='white')

# Simple arrows - just straight lines
arrows = [
    # Main flow
    ((6, 6.7), (6, 6.3)),      # User to FastAPI
    ((6, 5.7), (6, 5.3)),      # FastAPI to LangGraph
    ((5, 4.7), (3, 3.8)),      # LangGraph to Pinecone
    ((7, 4.7), (10, 3.8)),     # LangGraph to Tavily
    ((6, 4.3), (6, 2.3)),      # LangGraph to Response
    ((2, 3.2), (5, 2.3)),      # Pinecone to Response
    ((10, 3.2), (7, 2.3)),     # Tavily to Response
    ((6, 1.7), (6, 1.3))       # Response to Frontend
]

# Draw simple arrows
for start, end in arrows:
    arrow = FancyArrowPatch(
        start, end,
        arrowstyle='->',
        mutation_scale=15,
        color='black',
        linewidth=2
    )
    ax.add_patch(arrow)

# Add title
ax.text(6, 7.5, 'Mineral Insights - Simple Workflow', 
        ha='center', va='center', fontsize=14, fontweight='bold')

# Add info box at top right
info_text = """Components:
• User Query → FastAPI → LangGraph
• LangGraph → Pinecone (documents) + Tavily (web)
• All data → Response Assembly → Frontend"""

ax.text(9.5, 7, info_text, ha='left', va='top', 
        fontsize=9, 
        bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.8))

plt.tight_layout()
plt.savefig('mineral_insights_simple_workflow.png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ Simple workflow diagram with top-right info box generated: mineral_insights_simple_workflow.png")
