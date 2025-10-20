# Mineral Rights Chatbot Frontend Design Sketch

## Layout Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           MINERAL RIGHTS INSIGHTS                              │
│                              AI-Powered Assistant                              │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────────────────────┐  ┌─────────────────────────────────────┐   │
│  │                                 │  │                                     │   │
│  │         CHAT INTERFACE          │  │         INTERACTIVE MAP             │   │
│  │                                 │  │                                     │   │
│  │  ┌─────────────────────────────┐ │  │  ┌─────────────────────────────┐   │   │
│  │  │ 💬 Chat Messages            │ │  │  │ 🗺️  Map View                │   │   │
│  │  │                             │ │  │  │                             │   │   │
│  │  │ User: What are typical      │ │  │  │  ┌─────────────────────────┐ │   │   │
│  │  │ lease terms in Oklahoma?    │ │  │  │  │                         │ │   │   │
│  │  │                             │ │  │  │  │    Interactive Map      │ │   │   │
│  │  │ 🤖 Based on recent data,    │ │  │  │  │                         │ │   │   │
│  │  │ typical lease terms in      │ │  │  │  │  • Well locations       │ │   │   │
│  │  │ Oklahoma range from $500-   │ │  │  │  │  • Lease boundaries     │ │   │   │
│  │  │ $2,000 per acre bonus...    │ │  │  │  │  • County highlights    │ │   │   │
│  │  │                             │ │  │  │  │  • Operator activity    │ │   │   │
│  │  │ 📊 Confidence: 92%          │ │  │  │  │                         │ │   │   │
│  │  │ 📚 Sources: 5 documents     │ │  │  │  │                         │ │   │   │
│  │  │                             │ │  │  │  └─────────────────────────┘ │   │   │
│  │  └─────────────────────────────┘ │  │  │                             │   │   │
│  │                                 │  │  │  ┌─────────────────────────┐ │   │   │
│  │  ┌─────────────────────────────┐ │  │  │  │ 📍 Map Controls        │ │   │   │
│  │  │ Type your question...       │ │  │  │  │                         │ │   │   │
│  │  │ [Send] [📎] [🗺️] [📊]      │ │  │  │  │ • County Filter         │ │   │   │
│  │  └─────────────────────────────┘ │  │  │  │ • Operator Filter       │ │   │   │
│  │                                 │  │  │  │ • Formation Filter      │ │   │   │
│  │                                 │  │  │  │ • Date Range            │ │   │   │
│  │                                 │  │  │  │                         │ │   │   │
│  │                                 │  │  │  │ [Reset Filters]         │ │   │   │
│  │                                 │  │  │  └─────────────────────────┘ │   │   │
│  │                                 │  │  │                             │   │   │
│  │                                 │  │  │  ┌─────────────────────────┐ │   │   │
│  │                                 │  │  │  │ 📈 Data Insights        │ │   │   │
│  │                                 │  │  │  │                         │ │   │   │
│  │                                 │  │  │  │ • Active Wells: 1,247   │ │   │   │
│  │                                 │  │  │  │ • Avg Lease Price: $1.2K│ │   │   │
│  │                                 │  │  │  │ • Top Operator: Pioneer │ │   │   │
│  │                                 │  │  │  │ • Recent Activity: 23   │ │   │   │
│  │                                 │  │  │  │                         │ │   │   │
│  │                                 │  │  │  └─────────────────────────┘ │   │   │
│  │                                 │  │  └─────────────────────────────┘   │   │
│  └─────────────────────────────────┘  └─────────────────────────────────────┘   │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  📊 QUICK STATS  │  🔍 SEARCH FILTERS  │  📋 RECENT QUERIES  │  ⚙️ SETTINGS    │
│  • 2,847 Wells   │  • County: All      │  • OK lease terms   │  • Theme        │
│  • $1.2K Avg     │  • Operator: All    │  • TX permits       │  • Notifications│
│  • 156 Counties  │  • Formation: All   │  • Market trends    │  • Export Data  │
│  • 89 Operators  │  • Date: Last 30d   │  • Price analysis   │  • API Access   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Key Features

### 1. Chat Interface (Left Panel)
- **Message History**: Scrollable chat with user questions and AI responses
- **Confidence Indicators**: Visual confidence scores for each response
- **Source Citations**: Expandable source references
- **Input Area**: Rich text input with quick action buttons
- **Quick Actions**: 
  - 📎 Attach documents
  - 🗺️ Show on map
  - 📊 Generate chart
  - 💾 Save response

### 2. Interactive Map (Right Panel)
- **Map View**: Interactive map showing:
  - Well locations with drilling activity
  - Lease boundaries and offers
  - County-level data visualization
  - Operator activity heatmaps
- **Map Controls**: 
  - County/Operator/Formation filters
  - Date range selection
  - Layer toggles (wells, leases, permits)
- **Data Insights Panel**: Real-time statistics and key metrics

### 3. Bottom Navigation
- **Quick Stats**: Key metrics at a glance
- **Search Filters**: Global filter controls
- **Recent Queries**: Quick access to previous searches
- **Settings**: User preferences and data export options

## Visual Design Elements

### Color Scheme
- **Primary**: Deep blue (#1e3a8a) - Professional, trustworthy
- **Secondary**: Gold (#f59e0b) - Mineral/energy theme
- **Accent**: Green (#10b981) - Success/positive indicators
- **Background**: Light gray (#f8fafc) - Clean, modern
- **Text**: Dark gray (#1f2937) - High readability

### Typography
- **Headers**: Bold, sans-serif (Inter or Roboto)
- **Body**: Clean, readable sans-serif
- **Code/Data**: Monospace for technical information

### Interactive Elements
- **Hover Effects**: Subtle shadows and color changes
- **Loading States**: Skeleton screens and progress indicators
- **Responsive Design**: Mobile-friendly layout
- **Accessibility**: High contrast, keyboard navigation

## Technical Implementation

### Frontend Stack
- **Framework**: React.js with TypeScript
- **UI Library**: Tailwind CSS + Headless UI
- **Maps**: Mapbox GL JS or Leaflet
- **Charts**: Chart.js or D3.js
- **State Management**: Redux Toolkit or Zustand

### Backend Integration
- **API**: RESTful API connecting to your LangGraph workflow
- **Real-time**: WebSocket for live updates
- **Authentication**: JWT-based user management
- **Data**: Integration with your Qdrant vector store

### Key Components
1. **ChatContainer**: Main chat interface
2. **MessageBubble**: Individual chat messages
3. **MapView**: Interactive map component
4. **FilterPanel**: Data filtering controls
5. **StatsDashboard**: Key metrics display
6. **SourceModal**: Detailed source information

This design provides a comprehensive, professional interface for your mineral rights chatbot with powerful visualization capabilities.
