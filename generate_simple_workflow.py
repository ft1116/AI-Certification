#!/usr/bin/env python3
"""
Generate a simple PNG diagram of the Mineral Insights workflow
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

def generate_simple_workflow(filename="mineral_insights_simple_workflow.png"):
    fig, ax = plt.subplots(1, 1, figsize=(12, 14))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis('off')
    
    # Define colors
    colors = {
        'user': '#E6F3FF',
        'frontend': '#ADD8E6', 
        'backend': '#90EE90',
        'langgraph': '#FFD700',
        'agent': '#FFB6C1',
        'database': '#87CEEB',
        'api': '#F0E68C',
        'decision': '#FFD700'
    }
    
    # Define positions and boxes - SIMPLIFIED
    boxes = {
        'A': {'pos': (5, 13), 'size': (2, 0.8), 'text': 'User Query', 'color': colors['user'], 'shape': 'ellipse'},
        'B': {'pos': (5, 11), 'size': (2.5, 0.8), 'text': 'FastAPI Backend', 'color': colors['backend'], 'shape': 'rect'},
        'C': {'pos': (5, 9), 'size': (2.5, 0.8), 'text': 'LangGraph\nOrchestrator', 'color': colors['langgraph'], 'shape': 'rect'},
        'D': {'pos': (2, 7), 'size': (2, 0.8), 'text': 'Pinecone\nVector DB', 'color': colors['database'], 'shape': 'cylinder'},
        'E': {'pos': (8, 7), 'size': (2, 0.8), 'text': 'Mapping Agent', 'color': colors['agent'], 'shape': 'rect'},
        'F': {'pos': (8, 5), 'size': (2, 0.8), 'text': 'SQLite\nPermit DB', 'color': colors['database'], 'shape': 'cylinder'},
        'G': {'pos': (2, 5), 'size': (2, 0.8), 'text': 'Web Search\n(Tavily)', 'color': colors['api'], 'shape': 'oval'},
        'H': {'pos': (5, 3), 'size': (2.5, 0.8), 'text': 'Response\nAssembly', 'color': colors['backend'], 'shape': 'rect'},
        'I': {'pos': (5, 1), 'size': (2.5, 0.8), 'text': 'React Frontend', 'color': colors['frontend'], 'shape': 'rect'}
    }
    
    # Draw boxes
    for key, box in boxes.items():
        x, y = box['pos']
        w, h = box['size']
        
        if box['shape'] == 'ellipse':
            circle = patches.Ellipse((x, y), w, h, facecolor=box['color'], edgecolor='black', linewidth=2)
            ax.add_patch(circle)
        elif box['shape'] == 'cylinder':
            rect = FancyBboxPatch((x-w/2, y-h/2), w, h, 
                                boxstyle="round,pad=0.1", 
                                facecolor=box['color'], edgecolor='black', linewidth=2)
            ax.add_patch(rect)
        elif box['shape'] == 'oval':
            circle = patches.Ellipse((x, y), w, h, facecolor=box['color'], edgecolor='black', linewidth=2)
            ax.add_patch(circle)
        else:  # rect
            rect = FancyBboxPatch((x-w/2, y-h/2), w, h, 
                                boxstyle="round,pad=0.05", 
                                facecolor=box['color'], edgecolor='black', linewidth=2)
            ax.add_patch(rect)
        
        # Add text
        ax.text(x, y, box['text'], ha='center', va='center', fontsize=10, weight='bold')
    
    # Draw arrows - SIMPLIFIED
    arrows = [
        ('A', 'B', 'API Request'),
        ('B', 'C', 'Process Query'),
        ('C', 'D', 'Retrieve Documents'),
        ('B', 'E', 'Mapping Query?'),
        ('E', 'F', 'Query Permits'),
        ('C', 'G', 'Low Confidence'),
        ('D', 'H', 'Answer'),
        ('F', 'H', 'Mapping Data'),
        ('G', 'H', 'Web Results'),
        ('H', 'I', 'Final Response')
    ]
    
    for start, end, label in arrows:
        start_pos = boxes[start]['pos']
        end_pos = boxes[end]['pos']
        
        # Calculate arrow position
        start_x, start_y = start_pos[0], start_pos[1] - 0.4
        end_x, end_y = end_pos[0], end_pos[1] + 0.4
        
        # Draw arrow
        ax.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y),
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'))
        
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
    ax.text(5, 13.8, 'Mineral Insights - Simple Workflow', ha='center', va='top', 
           fontsize=16, weight='bold')
    
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
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Simple workflow diagram saved as: {filename}")
    print("📊 Simplified diagram shows:")
    print("   • User → FastAPI → LangGraph")
    print("   • LangGraph → Pinecone + Web Search")
    print("   • FastAPI → Mapping Agent → SQLite")
    print("   • All data → Response Assembly → Frontend")

if __name__ == "__main__":
    generate_simple_workflow()
