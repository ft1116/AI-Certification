#!/usr/bin/env python3
"""
Generate a visual sketch of the Mineral Rights Chatbot Frontend
"""

from PIL import Image, ImageDraw, ImageFont
import os

def create_frontend_sketch():
    """Create a visual sketch of the frontend design"""
    
    # Create a large canvas
    width, height = 1200, 800
    image = Image.new('RGB', (width, height), color='#f8fafc')
    draw = ImageDraw.Draw(image)
    
    # Define colors
    primary_blue = '#1e3a8a'
    secondary_gold = '#f59e0b'
    accent_green = '#10b981'
    text_dark = '#1f2937'
    border_gray = '#e5e7eb'
    
    # Try to use a system font, fallback to default
    try:
        title_font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 24)
        header_font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 18)
        body_font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 14)
        small_font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 12)
    except:
        title_font = ImageFont.load_default()
        header_font = ImageFont.load_default()
        body_font = ImageFont.load_default()
        small_font = ImageFont.load_default()
    
    # Header
    draw.rectangle([0, 0, width, 60], fill=primary_blue)
    draw.text((width//2 - 200, 20), "MINERAL RIGHTS INSIGHTS", fill='white', font=title_font)
    draw.text((width//2 - 150, 40), "AI-Powered Assistant", fill='white', font=body_font)
    
    # Main content area
    main_y = 80
    main_height = height - 200
    
    # Left panel - Chat Interface
    chat_width = width // 2 - 20
    chat_x = 20
    
    # Chat panel background
    draw.rectangle([chat_x, main_y, chat_x + chat_width, main_y + main_height], 
                   fill='white', outline=border_gray, width=2)
    
    # Chat header
    draw.rectangle([chat_x, main_y, chat_x + chat_width, main_y + 40], fill='#f3f4f6')
    draw.text((chat_x + 10, main_y + 10), "💬 Chat Interface", fill=text_dark, font=header_font)
    
    # Chat messages area
    messages_y = main_y + 50
    messages_height = main_height - 120
    
    # Sample chat messages
    draw.rectangle([chat_x + 10, messages_y, chat_x + chat_width - 10, messages_y + 60], 
                   fill='#e0f2fe', outline=border_gray)
    draw.text((chat_x + 15, messages_y + 10), "User: What are typical lease terms in Oklahoma?", 
              fill=text_dark, font=body_font)
    
    draw.rectangle([chat_x + 10, messages_y + 70, chat_x + chat_width - 10, messages_y + 140], 
                   fill='#f0f9ff', outline=border_gray)
    draw.text((chat_x + 15, messages_y + 80), "🤖 Based on recent data, typical lease terms", 
              fill=text_dark, font=body_font)
    draw.text((chat_x + 15, messages_y + 100), "in Oklahoma range from $500-$2,000 per acre...", 
              fill=text_dark, font=body_font)
    draw.text((chat_x + 15, messages_y + 120), "📊 Confidence: 92% | 📚 Sources: 5 documents", 
              fill=accent_green, font=small_font)
    
    # Input area
    input_y = main_y + main_height - 60
    draw.rectangle([chat_x + 10, input_y, chat_x + chat_width - 10, input_y + 40], 
                   fill='white', outline=border_gray)
    draw.text((chat_x + 15, input_y + 12), "Type your question...", fill='#9ca3af', font=body_font)
    
    # Action buttons
    button_x = chat_x + chat_width - 120
    draw.rectangle([button_x, input_y + 5, button_x + 25, input_y + 35], fill=primary_blue)
    draw.text((button_x + 5, input_y + 12), "📎", fill='white', font=body_font)
    
    draw.rectangle([button_x + 30, input_y + 5, button_x + 55, input_y + 35], fill=secondary_gold)
    draw.text((button_x + 35, input_y + 12), "🗺️", fill='white', font=body_font)
    
    draw.rectangle([button_x + 60, input_y + 5, button_x + 85, input_y + 35], fill=accent_green)
    draw.text((button_x + 65, input_y + 12), "📊", fill='white', font=body_font)
    
    # Right panel - Map Interface
    map_x = width // 2 + 10
    map_width = width // 2 - 30
    
    # Map panel background
    draw.rectangle([map_x, main_y, map_x + map_width, main_y + main_height], 
                   fill='white', outline=border_gray, width=2)
    
    # Map header
    draw.rectangle([map_x, main_y, map_x + map_width, main_y + 40], fill='#f3f4f6')
    draw.text((map_x + 10, main_y + 10), "🗺️ Interactive Map", fill=text_dark, font=header_font)
    
    # Map area
    map_area_y = main_y + 50
    map_area_height = main_height - 200
    
    # Map background (simplified representation)
    draw.rectangle([map_x + 10, map_area_y, map_x + map_width - 10, map_area_y + map_area_height], 
                   fill='#f0f9ff', outline=border_gray)
    
    # Map elements (simplified)
    # County boundaries
    for i in range(3):
        for j in range(3):
            x = map_x + 20 + i * 80
            y = map_area_y + 20 + j * 60
            draw.rectangle([x, y, x + 70, y + 50], fill='white', outline=border_gray)
            draw.text((x + 5, y + 5), f"County {i*3+j+1}", fill=text_dark, font=small_font)
    
    # Well locations (dots)
    well_positions = [(map_x + 50, map_area_y + 40), (map_x + 130, map_area_y + 100), 
                     (map_x + 210, map_area_y + 60), (map_x + 90, map_area_y + 140)]
    for x, y in well_positions:
        draw.ellipse([x-3, y-3, x+3, y+3], fill=accent_green)
    
    # Map controls
    controls_y = map_area_y + map_area_height + 10
    draw.rectangle([map_x + 10, controls_y, map_x + map_width - 10, controls_y + 80], 
                   fill='#f9fafb', outline=border_gray)
    draw.text((map_x + 15, controls_y + 5), "📍 Map Controls", fill=text_dark, font=body_font)
    draw.text((map_x + 15, controls_y + 25), "• County Filter", fill=text_dark, font=small_font)
    draw.text((map_x + 15, controls_y + 40), "• Operator Filter", fill=text_dark, font=small_font)
    draw.text((map_x + 15, controls_y + 55), "• Formation Filter", fill=text_dark, font=small_font)
    draw.text((map_x + 15, controls_y + 70), "• Date Range", fill=text_dark, font=small_font)
    
    # Data insights
    insights_y = controls_y + 90
    draw.rectangle([map_x + 10, insights_y, map_x + map_width - 10, insights_y + 60], 
                   fill='#f0fdf4', outline=border_gray)
    draw.text((map_x + 15, insights_y + 5), "📈 Data Insights", fill=text_dark, font=body_font)
    draw.text((map_x + 15, insights_y + 25), "• Active Wells: 1,247", fill=text_dark, font=small_font)
    draw.text((map_x + 15, insights_y + 40), "• Avg Lease Price: $1.2K", fill=text_dark, font=small_font)
    draw.text((map_x + 15, insights_y + 55), "• Top Operator: Pioneer", fill=text_dark, font=small_font)
    
    # Bottom navigation
    nav_y = height - 100
    nav_height = 80
    
    # Quick stats
    draw.rectangle([20, nav_y, 300, nav_y + nav_height], fill='white', outline=border_gray)
    draw.text((30, nav_y + 5), "📊 QUICK STATS", fill=text_dark, font=body_font)
    draw.text((30, nav_y + 25), "• 2,847 Wells", fill=text_dark, font=small_font)
    draw.text((30, nav_y + 40), "• $1.2K Avg Price", fill=text_dark, font=small_font)
    draw.text((30, nav_y + 55), "• 156 Counties", fill=text_dark, font=small_font)
    
    # Search filters
    draw.rectangle([320, nav_y, 600, nav_y + nav_height], fill='white', outline=border_gray)
    draw.text((330, nav_y + 5), "🔍 SEARCH FILTERS", fill=text_dark, font=body_font)
    draw.text((330, nav_y + 25), "• County: All", fill=text_dark, font=small_font)
    draw.text((330, nav_y + 40), "• Operator: All", fill=text_dark, font=small_font)
    draw.text((330, nav_y + 55), "• Formation: All", fill=text_dark, font=small_font)
    
    # Recent queries
    draw.rectangle([620, nav_y, 900, nav_y + nav_height], fill='white', outline=border_gray)
    draw.text((630, nav_y + 5), "📋 RECENT QUERIES", fill=text_dark, font=body_font)
    draw.text((630, nav_y + 25), "• OK lease terms", fill=text_dark, font=small_font)
    draw.text((630, nav_y + 40), "• TX permits", fill=text_dark, font=small_font)
    draw.text((630, nav_y + 55), "• Market trends", fill=text_dark, font=small_font)
    
    # Settings
    draw.rectangle([920, nav_y, 1180, nav_y + nav_height], fill='white', outline=border_gray)
    draw.text((930, nav_y + 5), "⚙️ SETTINGS", fill=text_dark, font=body_font)
    draw.text((930, nav_y + 25), "• Theme", fill=text_dark, font=small_font)
    draw.text((930, nav_y + 40), "• Notifications", fill=text_dark, font=small_font)
    draw.text((930, nav_y + 55), "• Export Data", fill=text_dark, font=small_font)
    
    return image

def main():
    """Generate and save the frontend sketch"""
    print("🎨 Generating Mineral Rights Chatbot Frontend Sketch...")
    
    try:
        # Create the sketch
        image = create_frontend_sketch()
        
        # Save the image
        output_path = "/Users/fmt116/Desktop/AI Certification/mineral_rights_frontend_sketch.png"
        image.save(output_path, "PNG", quality=95)
        
        print(f"✅ Frontend sketch saved to: {output_path}")
        print("📐 Image dimensions: 1200x800 pixels")
        print("🎯 Features included:")
        print("   • Chat interface with message history")
        print("   • Interactive map with well locations")
        print("   • Map controls and data insights")
        print("   • Quick stats and search filters")
        print("   • Recent queries and settings")
        
    except Exception as e:
        print(f"❌ Error generating sketch: {e}")
        print("💡 Make sure you have Pillow installed: pip install Pillow")

if __name__ == "__main__":
    main()

