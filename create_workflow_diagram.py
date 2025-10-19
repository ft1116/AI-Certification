#!/usr/bin/env python3
"""
Create a PNG diagram of the Mineral Insights LangGraph workflow
"""

try:
    from matplotlib.patches import Rectangle, FancyBboxPatch
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyArrowPatch
    import numpy as np
    
    def create_workflow_diagram():
        """Create a workflow diagram and save as PNG"""
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 16))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 20)
        ax.axis('off')
        
        # Colors
        colors = {
            'start_end': '#E1F5FE',      # Light blue
            'process': '#E3F2FD',        # Light blue
            'decision': '#FFEBEE',       # Light red
            'web_search': '#FFF3E0',     # Light orange
            'enhanced': '#E8F5E8',       # Light green
            'validate': '#F3E5F5'        # Light purple
        }
        
        # Define boxes
        boxes = [
            # (x, y, width, height, text, color, style)
            (3, 19, 4, 1, "START", colors['start_end'], 'ellipse'),
            (3, 17, 4, 1, "RETRIEVE\nDOCUMENTS\n\nSingle Semantic Search\n(Qdrant Vector DB)", colors['process'], 'rect'),
            (3, 15, 4, 1, "RANK\nDOCUMENTS\n\nSmart Relevance Scoring\n+ Deduplication", colors['process'], 'rect'),
            (3, 13, 4, 1, "GENERATE\nANSWER\n\nRich Context from:\n• Forum Discussions\n• Texas Permits\n• Oklahoma Permits\n• Lease Offers\n• Mineral Offers", colors['process'], 'rect'),
            (3, 11, 4, 1, "CONFIDENCE\nCHECK\n\nThreshold: < 0.6", colors['decision'], 'diamond'),
            (1, 9, 3, 1, "TAVILY\nWEB SEARCH", colors['web_search'], 'rect'),
            (6, 9, 3, 1, "VALIDATE\n& FORMAT", colors['validate'], 'rect'),
            (1, 7, 3, 1, "GENERATE\nENHANCED\nANSWER", colors['enhanced'], 'rect'),
            (3, 5, 4, 1, "END", colors['start_end'], 'ellipse'),
        ]
        
        # Draw boxes
        for x, y, w, h, text, color, style in boxes:
            if style == 'ellipse':
                circle = mpatches.Ellipse((x + w/2, y + h/2), w, h, 
                                        facecolor=color, edgecolor='black', linewidth=2)
                ax.add_patch(circle)
            elif style == 'diamond':
                diamond = mpatches.FancyBboxPatch((x, y), w, h, 
                                                boxstyle="round,pad=0.1", 
                                                facecolor=color, edgecolor='black', linewidth=2)
                ax.add_patch(diamond)
            else:  # rect
                rect = Rectangle((x, y), w, h, facecolor=color, edgecolor='black', linewidth=2)
                ax.add_patch(rect)
            
            # Add text
            ax.text(x + w/2, y + h/2, text, ha='center', va='center', 
                   fontsize=8, weight='bold', wrap=True)
        
        # Draw arrows
        arrows = [
            # (start_x, start_y, end_x, end_y)
            (5, 19, 5, 18),      # START to RETRIEVE
            (5, 17, 5, 16),      # RETRIEVE to RANK
            (5, 15, 5, 14),      # RANK to GENERATE
            (5, 13, 5, 12),      # GENERATE to CONFIDENCE
            (3, 11, 2.5, 10),    # CONFIDENCE to TAVILY (left)
            (7, 11, 7.5, 10),    # CONFIDENCE to VALIDATE (right)
            (2.5, 9, 2.5, 8),    # TAVILY to ENHANCED
            (2.5, 7, 5, 6),      # ENHANCED to END
            (7.5, 9, 5, 6),      # VALIDATE to END
        ]
        
        for start_x, start_y, end_x, end_y in arrows:
            arrow = FancyArrowPatch((start_x, start_y), (end_x, end_y),
                                  arrowstyle='->', mutation_scale=20, 
                                  color='black', linewidth=2)
            ax.add_patch(arrow)
        
        # Add labels for decision paths
        ax.text(1.5, 10.5, "Low\nConfidence", ha='center', va='center', 
               fontsize=8, style='italic', color='red')
        ax.text(8.5, 10.5, "High\nConfidence", ha='center', va='center', 
               fontsize=8, style='italic', color='green')
        
        # Add title
        ax.text(5, 21, "MINERAL INSIGHTS - LANGGRAPH WORKFLOW", 
               ha='center', va='center', fontsize=16, weight='bold')
        
        # Add data sources info
        ax.text(5, 3, "DATA SOURCES: Forum (3K) • Texas Permits (5K) • OK Permits (636) • Lease Offers (419) • Mineral Offers (145)\n\nPOWERED BY: LangGraph + OpenAI GPT-4 + Qdrant + Tavily", 
               ha='center', va='center', fontsize=10, 
               bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.7))
        
        # Save as PNG
        plt.tight_layout()
        plt.savefig('mineral_insights_workflow.png', dpi=300, bbox_inches='tight')
        print("✅ PNG diagram saved as 'mineral_insights_workflow.png'")
        
        return True
        
    if __name__ == "__main__":
        create_workflow_diagram()
        
except ImportError:
    print("❌ matplotlib not installed. Install with: pip install matplotlib")
    print("📋 Alternative: Use the Mermaid code in 'langgraph_workflow.mmd' with https://mermaid.live/")
