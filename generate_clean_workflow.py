#!/usr/bin/env python3
"""
Generate Clean Mineral Insights Workflow Diagram
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Set up the figure
fig, ax = plt.subplots(1, 1, figsize=(14, 10))
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis('off')

# Define colors
colors = {
    'user': '#ADD8E6',      # Light blue
    'frontend': '#ADD8E6',  # Light blue
    'backend': '#90EE90',   # Light green
    'langgraph': '#FFD700', # Gold
    'agent': '#FFB6C1',     # Light pink
    'database': '#87CEEB',  # Sky blue
    'api': '#F0E68C'        # Khaki
}

# Define components with positions and sizes
components = {
    'user': {'pos': (7, 9), 'size': (1.5, 0.8), 'text': 'User Query', 'color': colors['user'], 'shape': 'ellipse'},
    'fastapi': {'pos': (7, 7.5), 'size': (2, 0.8), 'text': 'FastAPI Backend', 'color': colors['backend'], 'shape': 'rect'},
    'langgraph': {'pos': (7, 6), 'size': (2, 0.8), 'text': 'LangGraph\nOrchestrator', 'color': colors['langgraph'], 'shape': 'rect'},
    'pinecone': {'pos': (3, 4.5), 'size': (1.8, 0.8), 'text': 'Pinecone\nVector DB', 'color': colors['database'], 'shape': 'cylinder'},
    'tavily': {'pos': (11, 4.5), 'size': (1.8, 0.8), 'text': 'Web Search\n(Tavily)', 'color': colors['api'], 'shape': 'oval'},
    'mapping_agent': {'pos': (11, 3), 'size': (1.8, 0.8), 'text': 'Mapping\nAgent', 'color': colors['agent'], 'shape': 'rect'},
    'sqlite': {'pos': (11, 1.5), 'size': (1.8, 0.8), 'text': 'SQLite\nPermit DB', 'color': colors['database'], 'shape': 'cylinder'},
    'response': {'pos': (7, 3), 'size': (2, 0.8), 'text': 'Response\nAssembly', 'color': colors['backend'], 'shape': 'rect'},
    'frontend': {'pos': (7, 1.5), 'size': (2, 0.8), 'text': 'React Frontend', 'color': colors['frontend'], 'shape': 'rect'}
}

# Draw components
for comp_id, comp in components.items():
    x, y = comp['pos']
    width, height = comp['size']
    
    if comp['shape'] == 'ellipse':
        # Draw ellipse
        ellipse = patches.Ellipse((x, y), width, height, 
                                facecolor=comp['color'], 
                                edgecolor='black', linewidth=1.5)
        ax.add_patch(ellipse)
    elif comp['shape'] == 'cylinder':
        # Draw cylinder (rectangle with top/bottom lines)
        rect = FancyBboxPatch((x-width/2, y-height/2), width, height,
                            boxstyle="round,pad=0.1",
                            facecolor=comp['color'],
                            edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        # Add cylinder lines
        ax.plot([x-width/2, x-width/2], [y-height/2-0.1, y-height/2+0.1], 'k-', linewidth=2)
        ax.plot([x+width/2, x+width/2], [y-height/2-0.1, y-height/2+0.1], 'k-', linewidth=2)
    elif comp['shape'] == 'oval':
        # Draw oval
        oval = patches.Ellipse((x, y), width, height*0.6,
                              facecolor=comp['color'],
                              edgecolor='black', linewidth=1.5)
        ax.add_patch(oval)
    else:  # rect
        # Draw rectangle
        rect = FancyBboxPatch((x-width/2, y-height/2), width, height,
                            boxstyle="round,pad=0.1",
                            facecolor=comp['color'],
                            edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
    
    # Add text
    ax.text(x, y, comp['text'], ha='center', va='center', 
            fontsize=9, fontweight='bold')

# Define arrows with start/end points and labels
arrows = [
    # Main flow
    ('user', 'fastapi', 'API Request'),
    ('fastapi', 'langgraph', 'Process Query'),
    ('langgraph', 'pinecone', 'Retrieve Documents'),
    ('langgraph', 'tavily', 'Low Confidence'),
    ('langgraph', 'response', 'Answer'),
    ('pinecone', 'response', 'Retrieved Docs'),
    ('tavily', 'response', 'Web Results'),
    ('fastapi', 'mapping_agent', 'Mapping Query?'),
    ('mapping_agent', 'sqlite', 'Query Permits'),
    ('sqlite', 'response', 'Mapping Data'),
    ('response', 'frontend', 'Final Response')
]

# Draw arrows with proper positioning
for start, end, label in arrows:
    start_pos = components[start]['pos']
    end_pos = components[end]['pos']
    
    # Calculate arrow start/end points (outside the shapes)
    dx = end_pos[0] - start_pos[0]
    dy = end_pos[1] - start_pos[1]
    length = np.sqrt(dx**2 + dy**2)
    
    # Normalize direction
    if length > 0:
        dx_norm = dx / length
        dy_norm = dy / length
    else:
        dx_norm = 0
        dy_norm = 0
    
    # Calculate start and end points (outside shapes)
    start_offset = 0.5  # Distance from shape edge
    end_offset = 0.5
    
    start_x = start_pos[0] + dx_norm * start_offset
    start_y = start_pos[1] + dy_norm * start_offset
    end_x = end_pos[0] - dx_norm * end_offset
    end_y = end_pos[1] - dy_norm * end_offset
    
    # Draw arrow
    arrow = FancyArrowPatch(
        (start_x, start_y), (end_x, end_y),
        arrowstyle='->',
        mutation_scale=20,
        color='black',
        linewidth=2
    )
    ax.add_patch(arrow)
    
    # Add label (only for key arrows)
    if label in ['API Request', 'Process Query', 'Mapping Query?', 'Final Response']:
        mid_x, mid_y = (start_x + end_x) / 2, (start_y + end_y) / 2
        ax.text(mid_x, mid_y + 0.3, label, ha='center', va='bottom', fontsize=8, 
                bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
    elif label == 'Low Confidence':
        # Position "Low Confidence" label further right for clarity
        mid_x, mid_y = (start_x + end_x) / 2 + 0.5, (start_y + end_y) / 2
        ax.text(mid_x, mid_y + 0.3, label, ha='center', va='bottom', fontsize=8, 
                bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))

# Add title
ax.text(7, 9.7, 'Mineral Insights - Simple Workflow', 
        ha='center', va='center', fontsize=16, fontweight='bold')

# Add legend
legend_elements = [
    patches.Patch(color=colors['user'], label='User'),
    patches.Patch(color=colors['frontend'], label='Frontend'),
    patches.Patch(color=colors['backend'], label='Backend'),
    patches.Patch(color=colors['langgraph'], label='LangGraph'),
    patches.Patch(color=colors['agent'], label='AI Agent'),
    patches.Patch(color=colors['database'], label='Database'),
    patches.Patch(color=colors['api'], label='External API')
]

ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1))

plt.tight_layout()
plt.savefig('mineral_insights_simple_workflow.png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ Clean workflow diagram generated: mineral_insights_simple_workflow.png")
