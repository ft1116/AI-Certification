#!/usr/bin/env python3
"""
Generate ACCURATE System Architecture Diagram for Mineral Insights
Based on actual implementation in chatbot.py and langgraph_chatbot.py
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

def create_accurate_architecture_diagram():
    """Create an accurate system architecture diagram based on real implementation"""
    
    # Create figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(16, 14))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis('off')
    
    # Define colors
    colors = {
        'frontend': '#E3F2FD',      # Light blue
        'backend': '#F3E5F5',       # Light purple
        'database': '#E8F5E8',      # Light green
        'external': '#FFF3E0',      # Light orange
        'ai': '#FCE4EC',            # Light pink
        'text': '#333333'           # Dark gray
    }
    
    # Title
    ax.text(5, 13.5, 'Mineral Insights: ACTUAL System Architecture', 
            fontsize=20, fontweight='bold', ha='center', color=colors['text'])
    
    # Frontend Layer
    frontend_box = FancyBboxPatch((0.5, 12), 9, 1.2, 
                                 boxstyle="round,pad=0.1", 
                                 facecolor=colors['frontend'], 
                                 edgecolor='#1976D2', linewidth=2)
    ax.add_patch(frontend_box)
    ax.text(5, 12.6, 'React Frontend', fontsize=14, fontweight='bold', ha='center')
    ax.text(5, 12.2, 'TypeScript • OpenLayers • Markdown Rendering', 
            fontsize=10, ha='center', va='center')
    
    # FastAPI Backend (chatbot.py)
    backend_box = FancyBboxPatch((1, 10.2), 8, 1.4, 
                                boxstyle="round,pad=0.1", 
                                facecolor=colors['backend'], 
                                edgecolor='#7B1FA2', linewidth=2)
    ax.add_patch(backend_box)
    ax.text(5, 10.9, 'FastAPI Backend (chatbot.py)', fontsize=12, fontweight='bold', ha='center')
    ax.text(5, 10.5, 'Location Extraction • Mapping Detection • Conversation Memory', 
            fontsize=9, ha='center')
    
    # LangGraph Orchestrator (langgraph_chatbot.py)
    orchestrator_box = FancyBboxPatch((1, 8.5), 8, 1.4, 
                                     boxstyle="round,pad=0.1", 
                                     facecolor=colors['ai'], 
                                     edgecolor='#C2185B', linewidth=2)
    ax.add_patch(orchestrator_box)
    ax.text(5, 9.2, 'LangGraph Orchestrator (langgraph_chatbot.py)', fontsize=12, fontweight='bold', ha='center')
    ax.text(5, 8.8, 'Document Retrieval • Ranking • Answer Generation • Confidence Scoring', 
            fontsize=9, ha='center')
    
    # Mapping Agent (mapping_agent.py)
    mapping_box = FancyBboxPatch((0.5, 6.8), 4, 1.2, 
                                boxstyle="round,pad=0.1", 
                                facecolor=colors['ai'], 
                                edgecolor='#C2185B', linewidth=2)
    ax.add_patch(mapping_box)
    ax.text(2.5, 7.4, 'Mapping Agent', fontsize=11, fontweight='bold', ha='center')
    ax.text(2.5, 7.0, 'Location Parsing • Permit Queries • GeoJSON Generation', 
            fontsize=8, ha='center')
    
    # Web Search Agent (Tavily)
    websearch_box = FancyBboxPatch((5.5, 6.8), 4, 1.2, 
                                  boxstyle="round,pad=0.1", 
                                  facecolor=colors['external'], 
                                  edgecolor='#F57C00', linewidth=2)
    ax.add_patch(websearch_box)
    ax.text(7.5, 7.4, 'Web Search Agent', fontsize=11, fontweight='bold', ha='center')
    ax.text(7.5, 7.0, 'Tavily API • Real-time Search • Confidence Triggered', 
            fontsize=8, ha='center')
    
    # Data Sources Layer
    data_y = 5.2
    data_width = 1.6
    data_height = 0.8
    
    data_sources = [
        ('Pinecone\nVector DB\n(27K docs)', 0.3),
        ('SQLite\nPermits DB\n(5.6K permits)', 2.2),
        ('Mineral Rights\nForum Data', 4.1),
        ('Texas RRC\nData', 5.9),
        ('Oklahoma\nOCC Data', 7.7)
    ]
    
    for source_name, x_pos in data_sources:
        data_box = FancyBboxPatch((x_pos, data_y), data_width, data_height,
                                 boxstyle="round,pad=0.05",
                                 facecolor=colors['database'],
                                 edgecolor='#388E3C', linewidth=1)
        ax.add_patch(data_box)
        ax.text(x_pos + data_width/2, data_y + data_height/2, source_name,
                fontsize=7, ha='center', va='center', fontweight='bold')
    
    # External APIs
    external_y = 3.5
    external_width = 1.8
    external_height = 0.8
    
    external_apis = [
        ('Claude\nSonnet 4.5', 1.5),
        ('OpenAI\nEmbeddings', 3.5),
        ('Geocoding\nServices', 5.5),
        ('Tavily\nSearch API', 7.5)
    ]
    
    for api_name, x_pos in external_apis:
        api_box = FancyBboxPatch((x_pos, external_y), external_width, external_height,
                                boxstyle="round,pad=0.05",
                                facecolor=colors['external'],
                                edgecolor='#F57C00', linewidth=1)
        ax.add_patch(api_box)
        ax.text(x_pos + external_width/2, external_y + external_height/2, api_name,
                fontsize=8, ha='center', va='center', fontweight='bold')
    
    # Actual Workflow Box
    workflow_box = FancyBboxPatch((0.2, 1.8), 9.6, 1.2,
                                 boxstyle="round,pad=0.1",
                                 facecolor='#F5F5F5',
                                 edgecolor='#666666', linewidth=1)
    ax.add_patch(workflow_box)
    ax.text(5, 2.4, 'Actual Workflow (No Query Classification)', fontsize=12, fontweight='bold', ha='center')
    ax.text(5, 2.0, '1. Location Extraction → 2. LangGraph Processing → 3. Mapping Detection → 4. Response Assembly', 
            fontsize=9, ha='center')
    
    # Key Features Box
    features_box = FancyBboxPatch((0.2, 0.2), 9.6, 1.2,
                                 boxstyle="round,pad=0.1",
                                 facecolor='#E8F5E8',
                                 edgecolor='#388E3C', linewidth=1)
    ax.add_patch(features_box)
    ax.text(5, 0.8, 'Key Features', fontsize=12, fontweight='bold', ha='center')
    ax.text(5, 0.4, '• Streaming Responses (SSE) • Confidence-Based Web Search • State Activity Weighting • Conversation Memory', 
            fontsize=9, ha='center')
    
    # Add connection arrows showing actual data flow
    # Frontend to Backend
    arrow1 = ConnectionPatch((5, 12), (5, 11.6), "data", "data",
                           arrowstyle="->", shrinkA=5, shrinkB=5,
                           mutation_scale=20, fc="black", linewidth=2)
    ax.add_patch(arrow1)
    
    # Backend to LangGraph
    arrow2 = ConnectionPatch((5, 10.2), (5, 9.9), "data", "data",
                           arrowstyle="->", shrinkA=5, shrinkB=5,
                           mutation_scale=20, fc="black", linewidth=2)
    ax.add_patch(arrow2)
    
    # Backend to Mapping Agent
    arrow3 = ConnectionPatch((3, 10.2), (2.5, 8), "data", "data",
                           arrowstyle="->", shrinkA=5, shrinkB=5,
                           mutation_scale=15, fc="black", linewidth=1)
    ax.add_patch(arrow3)
    
    # LangGraph to Web Search
    arrow4 = ConnectionPatch((7, 8.5), (7.5, 8), "data", "data",
                           arrowstyle="->", shrinkA=5, shrinkB=5,
                           mutation_scale=15, fc="black", linewidth=1)
    ax.add_patch(arrow4)
    
    # LangGraph to Pinecone
    arrow5 = ConnectionPatch((2, 8.5), (1.1, 6), "data", "data",
                           arrowstyle="->", shrinkA=5, shrinkB=5,
                           mutation_scale=15, fc="black", linewidth=1)
    ax.add_patch(arrow5)
    
    # Mapping Agent to SQLite
    arrow6 = ConnectionPatch((2.5, 6.8), (3.0, 6), "data", "data",
                           arrowstyle="->", shrinkA=5, shrinkB=5,
                           mutation_scale=15, fc="black", linewidth=1)
    ax.add_patch(arrow6)
    
    # LangGraph to Claude
    arrow7 = ConnectionPatch((6, 8.5), (2.4, 4.3), "data", "data",
                           arrowstyle="->", shrinkA=5, shrinkB=5,
                           mutation_scale=15, fc="black", linewidth=1)
    ax.add_patch(arrow7)
    
    # Add legend
    legend_elements = [
        patches.Patch(color=colors['frontend'], label='Frontend'),
        patches.Patch(color=colors['backend'], label='Backend (chatbot.py)'),
        patches.Patch(color=colors['ai'], label='AI Processing'),
        patches.Patch(color=colors['database'], label='Data Sources'),
        patches.Patch(color=colors['external'], label='External APIs')
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    plt.tight_layout()
    return fig

def main():
    """Generate and save the accurate architecture diagram"""
    print("🏗️ Generating ACCURATE Mineral Insights System Architecture Diagram...")
    
    fig = create_accurate_architecture_diagram()
    
    # Save as PNG
    output_file = "mineral_insights_accurate_architecture.png"
    fig.savefig(output_file, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    print(f"✅ ACCURATE Architecture diagram saved as: {output_file}")
    print("📊 Diagram shows ACTUAL implementation:")
    print("   • FastAPI Backend (chatbot.py) - Location extraction, mapping detection")
    print("   • LangGraph Orchestrator (langgraph_chatbot.py) - Document processing")
    print("   • Mapping Agent (mapping_agent.py) - Location parsing, permit queries")
    print("   • Web Search Agent - Tavily API integration")
    print("   • NO Query Classification - Confidence-based routing only")
    print("   • Actual data flow and component responsibilities")

if __name__ == "__main__":
    main()
