import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import { useLocation, LocationData } from './LocationContext';

interface Message {
  id: string;
  type: 'user' | 'assistant' | 'error';
  content: string;
  timestamp: Date;
  location?: LocationData;
}

const ChatAssistant: React.FC = () => {
  const [query, setQuery] = useState('');
  const [messages, setMessages] = useState<Message[]>([]);
  const [response, setResponse] = useState('');
  const [location, setLocation] = useState<LocationData | null>(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [streaming, setStreaming] = useState(false);
  const [useStreaming, setUseStreaming] = useState(true);
  const [conversationId, setConversationId] = useState(() => `conv_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`);
  const { setLocation: setMapLocation } = useLocation();
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Simple Markdown renderer function
  const renderMarkdown = (text: string) => {
    const lines = text.split('\n');
    const elements: React.ReactNode[] = [];
    let key = 0;

    lines.forEach((line) => {
      const trimmedLine = line.trim();
      
      if (!trimmedLine) {
        elements.push(<br key={key++} />);
        return;
      }

      // Headers
      if (trimmedLine.startsWith('### ')) {
        elements.push(
          <h3 key={key++} style={{ 
            fontSize: '1.1em', 
            fontWeight: 'bold', 
            margin: '12px 0 8px 0',
            color: 'white'
          }}>
            {trimmedLine.substring(4)}
          </h3>
        );
      } else if (trimmedLine.startsWith('## ')) {
        elements.push(
          <h2 key={key++} style={{ 
            fontSize: '1.3em', 
            fontWeight: 'bold', 
            margin: '16px 0 10px 0',
            color: 'white',
            borderBottom: '2px solid rgba(255, 255, 255, 0.2)',
            paddingBottom: '4px'
          }}>
            {trimmedLine.substring(3)}
          </h2>
        );
      } else if (trimmedLine.startsWith('# ')) {
        elements.push(
          <h1 key={key++} style={{ 
            fontSize: '1.5em', 
            fontWeight: 'bold', 
            margin: '20px 0 12px 0',
            color: 'white'
          }}>
            {trimmedLine.substring(2)}
          </h1>
        );
      }
      // Bullet points
      else if (trimmedLine.startsWith('- ')) {
        const bulletText = trimmedLine.substring(2);
        elements.push(
          <div key={key++} style={{ 
            margin: '4px 0 4px 20px',
            display: 'flex',
            alignItems: 'flex-start'
          }}>
            <span style={{ marginRight: '8px', color: 'rgba(255, 255, 255, 0.7)' }}>•</span>
            <span>{renderInlineMarkdown(bulletText)}</span>
          </div>
        );
      }
      // Regular paragraphs
      else {
        elements.push(
          <p key={key++} style={{ 
            margin: '8px 0',
            lineHeight: '1.6'
          }}>
            {renderInlineMarkdown(trimmedLine)}
          </p>
        );
      }
    });

    return elements;
  };

  const renderInlineMarkdown = (text: string) => {
    const parts = text.split(/(\*\*[^*]+\*\*)/g);
    
    return parts.map((part, index) => {
      if (part.startsWith('**') && part.endsWith('**')) {
        return (
          <strong key={index} style={{ fontWeight: 'bold', color: 'white' }}>
            {part.slice(2, -2)}
          </strong>
        );
      }
      return part;
    });
  };


  // Auto-scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, response]);

  const handleStreamingSubmit = async () => {
    if (!query.trim()) {
      setError('Please enter a query.');
      return;
    }
    
    const userQuery = query;
    
    // Add user message to history
    const userMessage: Message = {
      id: `user_${Date.now()}`,
      type: 'user',
      content: userQuery,
      timestamp: new Date()
    };
    setMessages(prev => [...prev, userMessage]);
    
    setStreaming(true);
    setLoading(true);
    setError('');
    setResponse('');
    setLocation(null);
    setQuery(''); // Clear input
    
    try {
      const response = await fetch('http://localhost:8003/chat/stream', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query: userQuery, conversation_id: conversationId }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();
      let fullResponse = '';
      let detectedLocation: LocationData | null = null;

      while (reader) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));
              
              if (data.type === 'location') {
                detectedLocation = data.data;
                setLocation(data.data);
                if (data.data) {
                  setMapLocation(data.data);
                }
              } else if (data.type === 'content') {
                fullResponse += data.data;
                setResponse(fullResponse);
              } else if (data.type === 'error') {
                setError(data.data);
              } else               if (data.type === 'done') {
                console.log('Stream completed');
                console.log('🔍 Done event data:', data);
                
                // Check if backend provided mapping data
                if (data.mapping_data) {
                  console.log('🗺️ Mapping data received from backend stream');
                  console.log(`🗺️ Found ${data.mapping_data.features?.length || 0} permits`);
                  const mapLocation = {
                    type: 'permits' as const,
                    data: data.mapping_data,
                    summary: data.mapping_summary
                  };
                  console.log('🗺️ Setting map location from stream:', mapLocation);
                  setMapLocation(mapLocation);
                } else {
                  console.log('❌ No mapping data in done event');
                }
              }
            } catch (e) {
              console.error('Error parsing stream data:', e);
            }
          }
        }
      }
      
      // Add assistant message to history after streaming completes
      if (fullResponse) {
        const assistantMessage: Message = {
          id: `assistant_${Date.now()}`,
          type: 'assistant',
          content: fullResponse,
          timestamp: new Date(),
          location: detectedLocation || undefined
        };
        setMessages(prev => [...prev, assistantMessage]);
        setResponse(''); // Clear temp response
      }
    } catch (err) {
      const errorMsg = 'Failed to connect to chatbot. Check server at http://localhost:8003.';
      setError(errorMsg);
      const errorMessage: Message = {
        id: `error_${Date.now()}`,
        type: 'error',
        content: errorMsg,
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
      console.error(err);
    } finally {
      setStreaming(false);
      setLoading(false);
    }
  };

  const handleRegularSubmit = async () => {
    if (!query.trim()) {
      setError('Please enter a query.');
      return;
    }
    
    const userQuery = query;
    
    // Add user message to history
    const userMessage: Message = {
      id: `user_${Date.now()}`,
      type: 'user',
      content: userQuery,
      timestamp: new Date()
    };
    setMessages(prev => [...prev, userMessage]);
    
    setLoading(true);
    setError('');
    setQuery(''); // Clear input
    
    try {
      const res = await axios.post('http://localhost:8003/chat', { query: userQuery }, {
        headers: { 'Content-Type': 'application/json' },
      });
      const responseText = res.data.answer || 'No response available.';
      setResponse(responseText);
      setLocation(res.data.location || null);
      
      // Add assistant message to history
      const assistantMessage: Message = {
        id: `assistant_${Date.now()}`,
        type: 'assistant',
        content: responseText,
        timestamp: new Date(),
        location: res.data.location || undefined
      };
      setMessages(prev => [...prev, assistantMessage]);
      setResponse(''); // Clear temp response
      
      // Set map location if backend found location data
      if (res.data.location) {
        console.log('Setting map location from backend:', res.data.location);
        console.log('Location type:', res.data.location.type);
        console.log('Location coordinates:', res.data.location.coordinates);
        setMapLocation(res.data.location);
        console.log('Map location set successfully');
      } else {
        console.log('No location data found in response');
      }
      
      // Check if this is a mapping query and trigger map display
      if (res.data.needs_mapping && res.data.mapping_data) {
        console.log('🗺️ Mapping data detected, triggering map display');
        console.log('🗺️ Mapping summary:', res.data.mapping_summary);
        console.log('🗺️ Mapping data features:', res.data.mapping_data.features?.length);
        // Trigger map display with permit data
        const mapLocation = {
          type: 'permits' as const,
          data: res.data.mapping_data,
          summary: res.data.mapping_summary
        };
        console.log('🗺️ Setting map location:', mapLocation);
        setMapLocation(mapLocation);
      }
    } catch (err) {
      const errorMsg = 'Failed to connect to chatbot. Check server at http://localhost:8003.';
      setError(errorMsg);
      const errorMessage: Message = {
        id: `error_${Date.now()}`,
        type: 'error',
        content: errorMsg,
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = async () => {
    if (useStreaming) {
      await handleStreamingSubmit();
    } else {
      await handleRegularSubmit();
    }
  };

  const suggestedQuestions = [
    "What are my mineral rights worth?",
    "How do lease agreements work?",
    "What legal protections do I have?",
    "Current market rates for royalties?"
  ];

  const handleSuggestedQuestion = (question: string) => {
    setQuery(question);
  };

  return (
    <>
      <style>
        {`
          @keyframes blink {
            0%, 50% { opacity: 1; }
            51%, 100% { opacity: 0; }
          }
          .suggested-btn {
            transition: all 0.2s ease;
          }
          .suggested-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.3);
          }
          /* Custom scrollbar styling */
          .chat-messages::-webkit-scrollbar {
            width: 8px;
          }
          .chat-messages::-webkit-scrollbar-track {
            background: rgba(15, 23, 42, 0.5);
            border-radius: 10px;
          }
          .chat-messages::-webkit-scrollbar-thumb {
            background: rgba(16, 185, 129, 0.5);
            border-radius: 10px;
          }
          .chat-messages::-webkit-scrollbar-thumb:hover {
            background: rgba(16, 185, 129, 0.7);
          }
        `}
      </style>
      <div style={{ display: 'flex', flexDirection: 'column', height: '100%', maxHeight: 'calc(100vh - 150px)' }}>
        {/* Header */}
        <div style={{
          background: 'linear-gradient(135deg, rgba(17, 94, 89, 0.4) 0%, rgba(19, 78, 74, 0.4) 100%)',
          padding: '24px',
          borderRadius: '16px',
          border: '1px solid rgba(16, 185, 129, 0.3)',
          boxShadow: '0 4px 6px rgba(0,0,0,0.2)',
          marginBottom: '20px'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '8px' }}>
            <div style={{
              width: '40px',
              height: '40px',
              backgroundColor: '#10b981',
              borderRadius: '50%',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontSize: '20px'
            }}>
              💬
            </div>
            <h2 style={{ fontSize: '24px', fontWeight: '700', color: 'white', margin: 0 }}>
              Mineral Rights Assistant
            </h2>
          </div>
          <p style={{ fontSize: '15px', color: 'rgba(255, 255, 255, 0.85)', margin: 0 }}>
            Ask about valuations, leases, rights, and regulations. Be as specific as possible and include information like County and State as well as section, township and range if possible.
          </p>
        </div>

        {/* Scrollable Messages Area */}
        <div className="chat-messages" style={{
          flex: 1,
          overflowY: 'auto',
          display: 'flex',
          flexDirection: 'column',
          gap: '16px',
          marginBottom: '20px',
          paddingRight: '8px'
        }}>
          {/* Initial Message */}
          <div style={{ 
            backgroundColor: 'rgba(16, 185, 129, 0.15)', 
            padding: '18px', 
            borderRadius: '12px',
            border: '1px solid rgba(16, 185, 129, 0.3)'
          }}>
            <p style={{ fontSize: '15px', color: 'white', margin: 0, marginBottom: '8px' }}>
              Hello! I'm your mineral rights assistant. I can help you understand your rights, property valuations, lease agreements, and regulatory requirements. What would you like to know?
            </p>
            <p style={{ fontSize: '12px', color: 'rgba(255, 255, 255, 0.6)', margin: 0 }}>
              {new Date().toLocaleTimeString()}
            </p>
          </div>

          {/* Message History */}
          {messages.map((msg) => (
            <div key={msg.id}>
              {msg.type === 'user' ? (
                <div style={{
                  backgroundColor: 'rgba(30, 41, 59, 0.6)',
                  padding: '16px',
                  borderRadius: '12px',
                  border: '1px solid rgba(255, 255, 255, 0.1)',
                  marginLeft: 'auto',
                  maxWidth: '80%'
                }}>
                  <p style={{ fontSize: '15px', color: 'white', margin: 0, lineHeight: '1.6' }}>
                    {msg.content}
                  </p>
                  <p style={{ fontSize: '11px', color: 'rgba(255, 255, 255, 0.5)', margin: '8px 0 0 0' }}>
                    {msg.timestamp.toLocaleTimeString()}
                  </p>
                </div>
              ) : msg.type === 'assistant' ? (
                <div>
                  <div style={{
                    backgroundColor: 'rgba(16, 185, 129, 0.15)',
                    padding: '16px',
                    borderRadius: '12px',
                    border: '1px solid rgba(16, 185, 129, 0.3)',
                    maxWidth: '85%'
                  }}>
                    <div style={{ fontSize: '15px', color: 'white', margin: 0, lineHeight: '1.6' }}>
                      {renderMarkdown(msg.content)}
                    </div>
                    <p style={{ fontSize: '11px', color: 'rgba(255, 255, 255, 0.5)', margin: '8px 0 0 0' }}>
                      {msg.timestamp.toLocaleTimeString()}
                    </p>
                  </div>
                  {msg.location && (
                    <div style={{
                      backgroundColor: 'rgba(16, 185, 129, 0.1)',
                      padding: '12px',
                      borderRadius: '8px',
                      border: '1px solid rgba(16, 185, 129, 0.2)',
                      marginTop: '8px',
                      maxWidth: '85%'
                    }}>
                      <p style={{ fontSize: '13px', fontWeight: '600', color: '#10b981', margin: '0 0 4px 0' }}>
                        📍 Location Detected
                      </p>
                      {msg.location.type === 'str' ? (
                        <p style={{ color: 'rgba(255, 255, 255, 0.8)', margin: 0, fontSize: '13px' }}>
                          Section: {msg.location.section}, Township: {msg.location.township}, Range: {msg.location.range}
                        </p>
                      ) : (
                        <p style={{ color: 'rgba(255, 255, 255, 0.8)', margin: 0, fontSize: '13px' }}>
                          {msg.location.name} ({msg.location.type})
                        </p>
                      )}
                    </div>
                  )}
                </div>
              ) : (
                <div style={{
                  backgroundColor: 'rgba(239, 68, 68, 0.15)',
                  padding: '14px',
                  borderRadius: '12px',
                  border: '1px solid rgba(239, 68, 68, 0.3)',
                  color: '#fca5a5'
                }}>
                  ⚠️ {msg.content}
                </div>
              )}
            </div>
          ))}

          {/* Current Streaming Response */}
          {response && streaming && (
            <div style={{
              backgroundColor: 'rgba(16, 185, 129, 0.15)',
              padding: '16px',
              borderRadius: '12px',
              border: '1px solid rgba(16, 185, 129, 0.3)',
              maxWidth: '85%'
            }}>
              <div style={{ fontSize: '15px', color: 'white', margin: 0, lineHeight: '1.6' }}>
                {renderMarkdown(response)}
                <span style={{ animation: 'blink 1s infinite' }}>|</span>
              </div>
              <p style={{ fontSize: '12px', color: 'rgba(16, 185, 129, 0.8)', marginTop: '8px', margin: '8px 0 0 0' }}>
                💬 Streaming response...
              </p>
            </div>
          )}

          {/* Scroll anchor */}
          <div ref={messagesEndRef} />
        </div>

        {/* Suggested Questions */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px' }}>
          {suggestedQuestions.map((question, idx) => (
            <button
              key={idx}
              className="suggested-btn"
              onClick={() => handleSuggestedQuestion(question)}
              disabled={loading}
              style={{
                padding: '12px 16px',
                backgroundColor: 'rgba(15, 23, 42, 0.6)',
                border: '1px solid rgba(16, 185, 129, 0.3)',
                borderRadius: '10px',
                color: 'rgba(255, 255, 255, 0.9)',
                fontSize: '13px',
                cursor: loading ? 'not-allowed' : 'pointer',
                textAlign: 'left',
                opacity: loading ? 0.5 : 1
              }}
            >
              {question}
            </button>
          ))}
        </div>

        {/* Input Area */}
        <div style={{ 
          display: 'flex', 
          gap: '10px',
          backgroundColor: 'rgba(15, 23, 42, 0.5)',
          padding: '16px',
          borderRadius: '16px',
          border: '1px solid rgba(16, 185, 129, 0.2)'
        }}>
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && !loading && handleSubmit()}
            placeholder="Ask about mineral rights, valuations, leases..."
            style={{
              flex: 1,
              padding: '14px 16px',
              backgroundColor: 'rgba(30, 41, 59, 0.6)',
              border: '1px solid rgba(255, 255, 255, 0.1)',
              borderRadius: '10px',
              color: 'white',
              fontSize: '14px',
              outline: 'none'
            }}
            disabled={loading}
          />
          <button
            onClick={handleSubmit}
            style={{
              backgroundColor: '#10b981',
              padding: '14px 24px',
              borderRadius: '10px',
              border: 'none',
              color: 'white',
              fontSize: '16px',
              cursor: loading ? 'not-allowed' : 'pointer',
              fontWeight: '600',
              boxShadow: '0 2px 4px rgba(0,0,0,0.2)',
              opacity: loading ? 0.6 : 1
            }}
            disabled={loading}
          >
            {loading ? '⏳' : '✈️'}
          </button>
        </div>

        {/* Controls */}
        <div style={{ 
          display: 'flex', 
          justifyContent: 'space-between', 
          alignItems: 'center',
          fontSize: '12px',
          color: 'rgba(255, 255, 255, 0.6)'
        }}>
          <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer' }}>
            <input
              type="checkbox"
              checked={useStreaming}
              onChange={(e) => setUseStreaming(e.target.checked)}
              style={{ margin: 0 }}
            />
            <span>Streaming {useStreaming ? 'ON' : 'OFF'}</span>
          </label>
          <button
            onClick={() => {
              setConversationId(`conv_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`);
              setMessages([]); // Clear message history
              setResponse('');
              setLocation(null);
              setError('');
            }}
            style={{
              backgroundColor: 'rgba(107, 114, 128, 0.3)',
              padding: '8px 14px',
              borderRadius: '8px',
              border: '1px solid rgba(255, 255, 255, 0.1)',
              color: 'rgba(255, 255, 255, 0.7)',
              cursor: 'pointer',
              fontSize: '12px',
              fontWeight: '600'
            }}
            title="Start a new conversation"
          >
            🔄 New Chat
          </button>
          <span style={{ fontSize: '11px' }}>ID: {conversationId.slice(-8)}</span>
        </div>
      </div>
    </>
  );
};

export default ChatAssistant;
