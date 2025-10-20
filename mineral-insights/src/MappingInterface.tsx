import React, { useState } from 'react';
import PermitsMap from './PermitsMap';

interface MappingInterfaceProps {
  onQuerySubmit?: (query: string) => void;
}

const MappingInterface: React.FC<MappingInterfaceProps> = ({ onQuerySubmit }) => {
  const [query, setQuery] = useState('');
  const [currentQuery, setCurrentQuery] = useState('');
  const [showMap, setShowMap] = useState(false);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (query.trim()) {
      setCurrentQuery(query.trim());
      setShowMap(true);
      if (onQuerySubmit) {
        onQuerySubmit(query.trim());
      }
    }
  };

  const exampleQueries = [
    "Show me all drilling permits in Grady County",
    "Permits in section 15 township 15n range 24w",
    "Oil wells by MEWBOURNE OIL COMPANY",
    "Horizontal wells targeting Woodford formation",
    "Recent permits in Roger Mills County",
    "All permits approved in September 2025"
  ];

  const handleExampleClick = (exampleQuery: string) => {
    setQuery(exampleQuery);
    setCurrentQuery(exampleQuery);
    setShowMap(true);
    if (onQuerySubmit) {
      onQuerySubmit(exampleQuery);
    }
  };

  return (
    <div style={{ 
      padding: '20px', 
      backgroundColor: '#f8f9fa', 
      borderRadius: '8px',
      margin: '20px 0'
    }}>
      <h2 style={{ 
        margin: '0 0 20px 0', 
        color: '#2c3e50',
        textAlign: 'center'
      }}>
        🗺️ Oklahoma Drilling Permits Map
      </h2>
      
      <form onSubmit={handleSubmit} style={{ marginBottom: '20px' }}>
        <div style={{ 
          display: 'flex', 
          gap: '10px', 
          marginBottom: '15px',
          alignItems: 'flex-start'
        }}>
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Enter location query (e.g., 'Show me permits in Grady County')"
            style={{
              flex: 1,
              padding: '12px',
              border: '2px solid #ddd',
              borderRadius: '6px',
              fontSize: '16px',
              outline: 'none',
              transition: 'border-color 0.3s'
            }}
            onFocus={(e) => e.target.style.borderColor = '#3498db'}
            onBlur={(e) => e.target.style.borderColor = '#ddd'}
          />
          <button
            type="submit"
            disabled={!query.trim()}
            style={{
              padding: '12px 24px',
              backgroundColor: query.trim() ? '#3498db' : '#bdc3c7',
              color: 'white',
              border: 'none',
              borderRadius: '6px',
              cursor: query.trim() ? 'pointer' : 'not-allowed',
              fontSize: '16px',
              fontWeight: 'bold',
              transition: 'background-color 0.3s'
            }}
          >
            Map Permits
          </button>
        </div>
      </form>

      <div style={{ marginBottom: '20px' }}>
        <h4 style={{ 
          margin: '0 0 10px 0', 
          color: '#34495e',
          fontSize: '14px'
        }}>
          💡 Try these example queries:
        </h4>
        <div style={{ 
          display: 'flex', 
          flexWrap: 'wrap', 
          gap: '8px' 
        }}>
          {exampleQueries.map((example, index) => (
            <button
              key={index}
              onClick={() => handleExampleClick(example)}
              style={{
                padding: '6px 12px',
                backgroundColor: '#ecf0f1',
                color: '#2c3e50',
                border: '1px solid #bdc3c7',
                borderRadius: '4px',
                cursor: 'pointer',
                fontSize: '12px',
                transition: 'all 0.3s'
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.backgroundColor = '#3498db';
                e.currentTarget.style.color = 'white';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.backgroundColor = '#ecf0f1';
                e.currentTarget.style.color = '#2c3e50';
              }}
            >
              {example}
            </button>
          ))}
        </div>
      </div>

      {showMap && currentQuery && (
        <div style={{ marginTop: '20px' }}>
          <div style={{ 
            display: 'flex', 
            justifyContent: 'space-between', 
            alignItems: 'center',
            marginBottom: '15px'
          }}>
            <h3 style={{ 
              margin: 0, 
              color: '#2c3e50',
              fontSize: '18px'
            }}>
              Map Results: "{currentQuery}"
            </h3>
            <button
              onClick={() => {
                setShowMap(false);
                setCurrentQuery('');
              }}
              style={{
                padding: '8px 16px',
                backgroundColor: '#e74c3c',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                cursor: 'pointer',
                fontSize: '14px'
              }}
            >
              Clear Map
            </button>
          </div>
          
          <PermitsMap query={currentQuery} />
        </div>
      )}

      <div style={{ 
        marginTop: '20px', 
        padding: '15px', 
        backgroundColor: '#e8f4fd',
        borderRadius: '6px',
        border: '1px solid #bee5eb'
      }}>
        <h4 style={{ 
          margin: '0 0 10px 0', 
          color: '#0c5460',
          fontSize: '14px'
        }}>
          📋 How to use the mapping agent:
        </h4>
        <ul style={{ 
          margin: 0, 
          paddingLeft: '20px', 
          color: '#0c5460',
          fontSize: '13px',
          lineHeight: '1.5'
        }}>
          <li><strong>County queries:</strong> "Show me permits in Grady County"</li>
          <li><strong>Specific locations:</strong> "Section 15, Township 15N, Range 24W"</li>
          <li><strong>Operator queries:</strong> "Permits by MEWBOURNE OIL COMPANY"</li>
          <li><strong>Formation targeting:</strong> "Wells targeting Woodford formation"</li>
          <li><strong>Well types:</strong> "Oil wells" or "Horizontal wells"</li>
          <li><strong>Date ranges:</strong> "Permits approved in September 2025"</li>
        </ul>
      </div>
    </div>
  );
};

export default MappingInterface;
