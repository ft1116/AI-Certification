
import React from 'react';
import MapConnect from './MapConnect';
import ChatAssistant from './ChatAssistant';

const App: React.FC = () => {
  return (
    <div style={{ 
      minHeight: '100vh', 
      background: 'linear-gradient(135deg, #0f766e 0%, #115e59 25%, #134e4a 50%, #78350f 75%, #92400e 100%)',
      color: 'white' 
    }}>
      {/* Header */}
      <header style={{ 
        padding: '32px 48px', 
        background: 'linear-gradient(135deg, rgba(15, 118, 110, 0.95) 0%, rgba(17, 94, 89, 0.95) 50%, rgba(120, 53, 15, 0.95) 100%)',
        backdropFilter: 'blur(10px)',
        display: 'flex', 
        justifyContent: 'space-between', 
        alignItems: 'center',
        borderBottom: '2px solid rgba(16, 185, 129, 0.3)'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          <div style={{
            width: '50px',
            height: '50px',
            backgroundColor: '#10b981',
            borderRadius: '12px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontSize: '28px',
            fontWeight: 'bold',
            color: 'white',
            boxShadow: '0 4px 6px rgba(0,0,0,0.3)'
          }}>
            ⛏️
          </div>
          <div>
            <h1 style={{ 
              fontSize: '32px', 
              fontWeight: '700', 
              color: 'white',
              margin: 0,
              letterSpacing: '-0.5px'
            }}>
              Mineral Rights Manager
            </h1>
            <p style={{ 
              fontSize: '15px', 
              color: 'rgba(255, 255, 255, 0.85)',
              margin: '4px 0 0 0'
            }}>
              Professional mineral rights management and consultation
            </p>
          </div>
        </div>
        <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '4px',
            padding: '8px 16px',
            backgroundColor: 'rgba(255, 255, 255, 0.15)',
            borderRadius: '8px',
            backdropFilter: 'blur(10px)',
            border: '1px solid rgba(255, 255, 255, 0.2)'
          }}>
            <span style={{ fontSize: '14px', color: 'rgba(255, 255, 255, 0.7)' }}>📍</span>
            <span style={{ fontSize: '14px', fontWeight: '600', color: 'white' }}>Property Overview</span>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main style={{ display: 'flex', minHeight: 'calc(100vh - 130px)' }}>
        {/* Left Panel - Map */}
        <div style={{ 
          width: '50%', 
          padding: '32px',
          backgroundColor: 'rgba(15, 23, 42, 0.3)',
          backdropFilter: 'blur(10px)'
        }}>
          <MapConnect />
        </div>

        {/* Right Panel - Chat */}
        <div style={{ 
          width: '50%', 
          padding: '32px',
          backgroundColor: 'rgba(30, 41, 59, 0.4)',
          backdropFilter: 'blur(10px)',
          borderLeft: '1px solid rgba(255, 255, 255, 0.1)'
        }}>
          <ChatAssistant />
        </div>
      </main>
    </div>
  );
};

export default App;

