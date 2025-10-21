#!/usr/bin/env python3
"""
Generate a PNG diagram of the Mineral Insights workflow using matplotlib
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

def generate_workflow_diagram(filename="mineral_insights_workflow.png"):
    fig, ax = plt.subplots(1, 1, figsize=(16, 20))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 20)
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
        'decision': '#FFD700',
        'final': '#98FB98'
    }
    
    # Define positions and boxes
    boxes = {
        'A': {'pos': (5, 19), 'size': (1.5, 0.8), 'text': 'User Query', 'color': colors['user'], 'shape': 'ellipse'},
        'B': {'pos': (5, 17), 'size': (2, 0.8), 'text': 'FastAPI Backend', 'color': colors['backend'], 'shape': 'rect'},
        'C': {'pos': (2, 15.5), 'size': (1.8, 0.6), 'text': 'Location\nExtraction', 'color': colors['backend'], 'shape': 'rect'},
        'D': {'pos': (5, 15.5), 'size': (2, 0.6), 'text': 'LangGraph\nOrchestrator', 'color': colors['langgraph'], 'shape': 'rect'},
        'E': {'pos': (2, 14), 'size': (1.8, 0.6), 'text': 'Document\nRetrieval', 'color': colors['langgraph'], 'shape': 'rect'},
        'F': {'pos': (0.5, 12.5), 'size': (1.8, 0.6), 'text': 'Pinecone\nVector DB', 'color': colors['database'], 'shape': 'cylinder'},
        'G': {'pos': (4, 12.5), 'size': (1.8, 0.6), 'text': 'Document\nRanking', 'color': colors['langgraph'], 'shape': 'rect'},
        'H': {'pos': (7, 12.5), 'size': (1.8, 0.6), 'text': 'Answer\nGeneration', 'color': colors['langgraph'], 'shape': 'rect'},
        'I': {'pos': (7, 10.5), 'size': (1.8, 0.6), 'text': 'Confidence\nCheck', 'color': colors['decision'], 'shape': 'diamond'},
        'J': {'pos': (5, 9), 'size': (1.8, 0.6), 'text': 'Web Search\nAgent', 'color': colors['agent'], 'shape': 'rect'},
        'K': {'pos': (2, 7.5), 'size': (1.5, 0.6), 'text': 'Tavily API', 'color': colors['api'], 'shape': 'oval'},
        'L': {'pos': (5, 6), 'size': (2, 0.6), 'text': 'Enhanced Answer\nGeneration', 'color': colors['langgraph'], 'shape': 'rect'},
        'M': {'pos': (7, 4.5), 'size': (1.5, 0.6), 'text': 'Final Answer', 'color': colors['final'], 'shape': 'rect'},
        'N': {'pos': (8.5, 15.5), 'size': (1.5, 0.6), 'text': 'Mapping\nQuery?', 'color': colors['decision'], 'shape': 'diamond'},
        'O': {'pos': (8.5, 14), 'size': (1.5, 0.6), 'text': 'Mapping\nAgent', 'color': colors['agent'], 'shape': 'rect'},
        'P': {'pos': (8.5, 12.5), 'size': (1.5, 0.6), 'text': 'SQLite\nPermit DB', 'color': colors['database'], 'shape': 'cylinder'},
        'Q': {'pos': (5, 3), 'size': (2, 0.6), 'text': 'Response\nAssembly', 'color': colors['backend'], 'shape': 'rect'},
        'R': {'pos': (5, 1), 'size': (2, 0.6), 'text': 'React Frontend', 'color': colors['frontend'], 'shape': 'rect'}
    }
    
    # Draw boxes
    for key, box in boxes.items():
        x, y = box['pos']
        w, h = box['size']
        
        if box['shape'] == 'ellipse':
            circle = patches.Ellipse((x, y), w, h, facecolor=box['color'], edgecolor='black', linewidth=1)
            ax.add_patch(circle)
        elif box['shape'] == 'diamond':
            diamond = patches.Polygon([(x, y+h/2), (x+w/2, y), (x, y-h/2), (x-w/2, y)], 
                                    facecolor=box['color'], edgecolor='black', linewidth=1)
            ax.add_patch(diamond)
        elif box['shape'] == 'cylinder':
            rect = FancyBboxPatch((x-w/2, y-h/2), w, h, 
                                boxstyle="round,pad=0.1", 
                                facecolor=box['color'], edgecolor='black', linewidth=1)
            ax.add_patch(rect)
        else:  # rect
            rect = FancyBboxPatch((x-w/2, y-h/2), w, h, 
                                boxstyle="round,pad=0.05", 
                                facecolor=box['color'], edgecolor='black', linewidth=1)
            ax.add_patch(rect)
        
        # Add text
        ax.text(x, y, box['text'], ha='center', va='center', fontsize=8, weight='bold')
    
    # Draw arrows
    arrows = [
        ('A', 'B', 'API Request'),
        ('B', 'C', 'Extract Location'),
        ('B', 'D', 'Process Query'),
        ('D', 'E', 'Retrieve Documents'),
        ('E', 'F', 'Query Vector DB'),
        ('D', 'G', 'Rank Documents'),
        ('D', 'H', 'Generate Answer'),
        ('H', 'I', 'Check Confidence'),
        ('I', 'J', 'Low Confidence'),
        ('J', 'K', 'Web Search'),
        ('K', 'L', 'Enhance Answer'),
        ('I', 'M', 'High Confidence'),
        ('L', 'M', 'Final Answer'),
        ('B', 'N', 'Check for Mapping'),
        ('N', 'O', 'Yes'),
        ('O', 'P', 'Query Permits'),
        ('M', 'Q', 'Assemble Response'),
        ('P', 'Q', 'Add Mapping Data'),
        ('Q', 'R', 'Return to Frontend')
    ]
    
    for start, end, label in arrows:
        start_pos = boxes[start]['pos']
        end_pos = boxes[end]['pos']
        
        # Calculate arrow position
        if start == 'I' and end == 'J':  # Low confidence arrow
            start_x, start_y = start_pos[0] - 0.5, start_pos[1] - 0.3
            end_x, end_y = end_pos[0], end_pos[1] + 0.3
        elif start == 'I' and end == 'M':  # High confidence arrow
            start_x, start_y = start_pos[0] + 0.5, start_pos[1] - 0.3
            end_x, end_y = end_pos[0], end_pos[1] + 0.3
        else:
            start_x, start_y = start_pos[0], start_pos[1] - 0.3
            end_x, end_y = end_pos[0], end_pos[1] + 0.3
        
        # Draw arrow
        ax.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y),
                   arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
        
        # Add label
        mid_x, mid_y = (start_x + end_x) / 2, (start_y + end_y) / 2
        ax.text(mid_x, mid_y + 0.2, label, ha='center', va='bottom', fontsize=6, 
               bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
    
    # Add title
    ax.text(5, 19.8, 'Mineral Insights System Workflow', ha='center', va='top', 
           fontsize=16, weight='bold')
    
    # Add legend
    legend_elements = [
        patches.Patch(color=colors['user'], label='User Interface'),
        patches.Patch(color=colors['frontend'], label='Frontend'),
        patches.Patch(color=colors['backend'], label='Backend'),
        patches.Patch(color=colors['langgraph'], label='LangGraph'),
        patches.Patch(color=colors['agent'], label='AI Agents'),
        patches.Patch(color=colors['database'], label='Database'),
        patches.Patch(color=colors['api'], label='External API'),
        patches.Patch(color=colors['decision'], label='Decision Point'),
        patches.Patch(color=colors['final'], label='Final Output')
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
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
