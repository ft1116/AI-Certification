#!/usr/bin/env python3
"""
Generate System Architecture Diagram for Mineral Insights
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

def create_system_architecture_diagram():
    """Create a comprehensive system architecture diagram"""
    
    # Create figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
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
    ax.text(5, 11.5, 'Mineral Insights: System Architecture', 
            fontsize=20, fontweight='bold', ha='center', color=colors['text'])
    
    # Frontend Layer
    frontend_box = FancyBboxPatch((0.5, 9.5), 9, 1.5, 
                                 boxstyle="round,pad=0.1", 
                                 facecolor=colors['frontend'], 
                                 edgecolor='#1976D2', linewidth=2)
    ax.add_patch(frontend_box)
    ax.text(5, 10.25, 'Frontend Layer', fontsize=14, fontweight='bold', ha='center')
    ax.text(5, 9.9, 'React + TypeScript\nOpenLayers Mapping\nMarkdown Rendering', 
            fontsize=10, ha='center', va='center')
    
    # API Gateway
    api_box = FancyBboxPatch((4, 8), 2, 0.8, 
                            boxstyle="round,pad=0.05", 
                            facecolor=colors['backend'], 
                            edgecolor='#7B1FA2', linewidth=2)
    ax.add_patch(api_box)
    ax.text(5, 8.4, 'FastAPI\nBackend', fontsize=10, fontweight='bold', ha='center')
    
    # LangGraph Orchestrator
    orchestrator_box = FancyBboxPatch((1, 6.5), 8, 1.2, 
                                     boxstyle="round,pad=0.1", 
                                     facecolor=colors['ai'], 
                                     edgecolor='#C2185B', linewidth=2)
    ax.add_patch(orchestrator_box)
    ax.text(5, 7.1, 'LangGraph Orchestrator', fontsize=12, fontweight='bold', ha='center')
    ax.text(5, 6.7, 'Query Classification • State Management • Workflow Control', 
            fontsize=9, ha='center')
    
    # AI Agents Layer
    agents_y = 5.2
    agent_width = 1.8
    agent_height = 0.8
    
    agents = [
        ('Document\nRetrieval\nAgent', 0.5),
        ('Mapping\nAgent', 2.5),
        ('Web Search\nAgent', 4.5),
        ('Response\nGeneration\nAgent', 6.5),
        ('Location\nParser', 8.5)
    ]
    
    for agent_name, x_pos in agents:
        agent_box = FancyBboxPatch((x_pos, agents_y), agent_width, agent_height,
                                  boxstyle="round,pad=0.05",
                                  facecolor=colors['ai'],
                                  edgecolor='#C2185B', linewidth=1)
        ax.add_patch(agent_box)
        ax.text(x_pos + agent_width/2, agents_y + agent_height/2, agent_name,
                fontsize=8, ha='center', va='center', fontweight='bold')
    
    # Data Sources Layer
    data_y = 3.5
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
    external_y = 2
    external_width = 1.8
    external_height = 0.8
    
    external_apis = [
        ('Tavily\nSearch API', 1.5),
        ('Geocoding\nServices', 3.5),
        ('Claude\nSonnet 4.5', 5.5),
        ('OpenAI\nEmbeddings', 7.5)
    ]
    
    for api_name, x_pos in external_apis:
        api_box = FancyBboxPatch((x_pos, external_y), external_width, external_height,
                                boxstyle="round,pad=0.05",
                                facecolor=colors['external'],
                                edgecolor='#F57C00', linewidth=1)
        ax.add_patch(api_box)
        ax.text(x_pos + external_width/2, external_y + external_height/2, api_name,
                fontsize=8, ha='center', va='center', fontweight='bold')
    
    # Key Features Box
    features_box = FancyBboxPatch((0.2, 0.2), 9.6, 1.2,
                                 boxstyle="round,pad=0.1",
                                 facecolor='#F5F5F5',
                                 edgecolor='#666666', linewidth=1)
    ax.add_patch(features_box)
    ax.text(5, 0.8, 'Key Features', fontsize=12, fontweight='bold', ha='center')
    ax.text(5, 0.4, '• Streaming Responses (SSE) • Interactive Mapping • Conversation Memory • Confidence Scoring • State Activity Weighting', 
            fontsize=9, ha='center')
    
    # Add connection arrows
    # Frontend to API
    arrow1 = ConnectionPatch((5, 9.5), (5, 8.8), "data", "data",
                           arrowstyle="->", shrinkA=5, shrinkB=5,
                           mutation_scale=20, fc="black", linewidth=2)
    ax.add_patch(arrow1)
    
    # API to Orchestrator
    arrow2 = ConnectionPatch((5, 8), (5, 7.7), "data", "data",
                           arrowstyle="->", shrinkA=5, shrinkB=5,
                           mutation_scale=20, fc="black", linewidth=2)
    ax.add_patch(arrow2)
    
    # Orchestrator to Agents
    for x_pos in [1.4, 3.4, 5.4, 7.4, 9.4]:
        arrow = ConnectionPatch((x_pos, 6.5), (x_pos, 6), "data", "data",
                              arrowstyle="->", shrinkA=5, shrinkB=5,
                              mutation_scale=15, fc="black", linewidth=1)
        ax.add_patch(arrow)
    
    # Agents to Data Sources
    for x_pos in [1.4, 3.4, 5.4, 7.4, 9.4]:
        arrow = ConnectionPatch((x_pos, 5.2), (x_pos, 4.3), "data", "data",
                              arrowstyle="->", shrinkA=5, shrinkB=5,
                              mutation_scale=15, fc="black", linewidth=1)
        ax.add_patch(arrow)
    
    # Data Sources to External APIs
    for x_pos in [1.1, 3.0, 5.0, 7.0, 8.5]:
        arrow = ConnectionPatch((x_pos, 3.5), (x_pos, 2.8), "data", "data",
                              arrowstyle="->", shrinkA=5, shrinkB=5,
                              mutation_scale=15, fc="black", linewidth=1)
        ax.add_patch(arrow)
    
    # Add legend
    legend_elements = [
        patches.Patch(color=colors['frontend'], label='Frontend'),
        patches.Patch(color=colors['backend'], label='Backend'),
        patches.Patch(color=colors['ai'], label='AI/Agents'),
        patches.Patch(color=colors['database'], label='Data Sources'),
        patches.Patch(color=colors['external'], label='External APIs')
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    plt.tight_layout()
    return fig

def main():
    """Generate and save the architecture diagram"""
    print("🏗️ Generating Mineral Insights System Architecture Diagram...")
    
    fig = create_system_architecture_diagram()
    
    # Save as PNG
    output_file = "mineral_insights_architecture.png"
    fig.savefig(output_file, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    print(f"✅ Architecture diagram saved as: {output_file}")
    print("📊 Diagram includes:")
    print("   • Frontend Layer (React + TypeScript)")
    print("   • FastAPI Backend")
    print("   • LangGraph Orchestrator")
    print("   • 5 Specialized AI Agents")
    print("   • 5 Data Sources (27K+ documents)")
    print("   • 4 External APIs")
    print("   • Key Features and Connections")

if __name__ == "__main__":
    main()
