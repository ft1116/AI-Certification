import React, { useState, useEffect } from 'react';
import PermitsMap from './PermitsMap';
import MapConnect from './MapConnect';
import { useLocation } from './LocationContext';

interface LeasingDashboardProps {
  onBack?: () => void;
}

const LeasingDashboard: React.FC<LeasingDashboardProps> = ({ onBack }) => {
  const [query, setQuery] = useState('');
  const [currentQuery, setCurrentQuery] = useState('');
  const [showMap, setShowMap] = useState(false);
  const [countyInput, setCountyInput] = useState('');
  const [selectedCounty, setSelectedCounty] = useState('');
  const [dashboardData, setDashboardData] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [permitCount, setPermitCount] = useState<number>(0);
  const { setLocation } = useLocation();

  const fetchDashboardData = async (county: string) => {
    if (!county.trim()) return;
    
    setLoading(true);
    setSelectedCounty(county);
    setCurrentQuery(`Show permits in ${county}`);
    setShowMap(true);
    
    try {
      // Fetch location data from chatbot to get coordinates
      const locationResponse = await fetch('http://localhost:8003/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: `Show me ${county}`,
          conversation_id: `location_${Date.now()}`
        })
      });
      
      const locationData = await locationResponse.json();
      
      // Set the map location using the context
      if (locationData.location) {
        setLocation(locationData.location);
      }
      
      // Fetch permits count
      const countyClean = county.replace(/\s+county/i, '').trim();
      const permitsResponse = await fetch(`http://localhost:8003/permits/location?county=${encodeURIComponent(countyClean)}&limit=500`);
      const permitsData = await permitsResponse.json();
      setPermitCount(permitsData.count || 0);
      
      // Fetch lease offers from forum
      const leaseQuery = `What are recent lease offers reported in ${county}? Include specific per-acre prices and royalty rates mentioned in the mineral rights forum discussions.`;
      
      const leaseResponse = await fetch('http://localhost:8003/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: leaseQuery,
          conversation_id: `lease_${Date.now()}`
        })
      });
      
      const leaseData = await leaseResponse.json();
      
      // Fetch AI analysis
      const analysisQuery = `Provide a detailed analysis for ${county}. Include: 1) Top formations being drilled with well counts, 2) Market trends and activity summary, 3) Operator activity and any notable developments. Format your response with clear sections.`;
      
      const response = await fetch('http://localhost:8003/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: analysisQuery,
          conversation_id: `dashboard_${Date.now()}`
        })
      });
      
      const data = await response.json();
      
      // Parse formations from permits data
      const formations = new Map<string, number>();
      if (permitsData.permits) {
        permitsData.permits.forEach((permit: any) => {
          // Extract formation from permit data if available
          const formation = permit.target_formation || 'Various Formations';
          formations.set(formation, (formations.get(formation) || 0) + 1);
        });
      }
      
      const formationsList = Array.from(formations.entries())
        .map(([name, count]) => ({ name, count }))
        .sort((a, b) => b.count - a.count)
        .slice(0, 6);
      
      setDashboardData({
        county: county,
        permitCount: permitsData.count || 0,
        formations: formationsList.length > 0 ? formationsList : [
          { name: 'Analysis in progress...', count: 0 }
        ],
        leaseOffers: leaseData.answer || 'Loading lease offer data from mineral rights forum...',
        marketSummary: data.answer || 'Loading market analysis...',
        hasData: permitsData.count > 0
      });
    } catch (error) {
      console.error('Error fetching dashboard data:', error);
      setDashboardData({
        county: county,
        permitCount: 0,
        formations: [],
        leaseOffers: 'Unable to load lease data at this time.',
        marketSummary: 'Unable to load market data at this time. Please try again.',
        hasData: false
      });
    } finally {
      setLoading(false);
    }
  };

  const handlePermitSearch = (e: React.FormEvent) => {
    e.preventDefault();
    if (query.trim()) {
      setCurrentQuery(query.trim());
      setShowMap(true);
    }
  };

  const exampleQueries = [
    "Show me drilling permits in Grady County Oklahoma",
    "Permits in DeWitt County Texas",
    "Recent permits in Karnes County",
    "Horizontal wells in Live Oak County Texas",
    "Permits in section 15 township 15n range 24w Oklahoma"
  ];

  const handleCountySubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (countyInput.trim()) {
      fetchDashboardData(countyInput.trim());
    }
  };

  const exampleCounties = [
    'Grady County, Oklahoma',
    'DeWitt County, Texas',
    'Karnes County, Texas',
    'Canadian County, Oklahoma',
    'Live Oak County, Texas'
  ];

  return (
    <div style={{ padding: '20px', maxWidth: '1400px', margin: '0 auto', backgroundColor: '#f5f5f5', minHeight: '100vh' }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '30px' }}>
        {onBack && (
          <button
            onClick={onBack}
            style={{
              padding: '10px 20px',
              backgroundColor: '#6c757d',
              color: 'white',
              border: 'none',
              borderRadius: '6px',
              cursor: 'pointer',
              fontSize: '14px',
              fontWeight: '500'
            }}
          >
            ← Back
          </button>
        )}
        <h1 style={{ 
          textAlign: 'center', 
          color: '#1e293b', 
          flex: 1, 
          margin: 0,
          fontSize: '32px',
          fontWeight: '700'
        }}>
          📊 County Activity Dashboard
        </h1>
        <div style={{ width: '80px' }}></div>
      </div>

      {/* County Input Bar - Full Width */}
      <div style={{
        backgroundColor: 'white',
        padding: '20px 30px',
        borderRadius: '12px',
        boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
        marginBottom: '20px'
      }}>
        <form onSubmit={handleCountySubmit}>
          <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
            <label style={{ fontSize: '16px', fontWeight: '600', color: '#1e293b', whiteSpace: 'nowrap' }}>
              🔍 County:
            </label>
            <input
              type="text"
              value={countyInput}
              onChange={(e) => setCountyInput(e.target.value)}
              placeholder="e.g., Grady County Oklahoma, DeWitt County Texas"
              style={{
                flex: 1,
                padding: '12px 20px',
                fontSize: '15px',
                border: '2px solid #e2e8f0',
                borderRadius: '8px',
                outline: 'none',
                transition: 'border-color 0.2s'
              }}
              onFocus={(e) => e.target.style.borderColor = '#3b82f6'}
              onBlur={(e) => e.target.style.borderColor = '#e2e8f0'}
            />
            <button
              type="submit"
              disabled={loading}
              style={{
                padding: '12px 32px',
                backgroundColor: loading ? '#9ca3af' : '#3b82f6',
                color: 'white',
                border: 'none',
                borderRadius: '8px',
                fontSize: '15px',
                fontWeight: '600',
                cursor: loading ? 'not-allowed' : 'pointer',
                transition: 'background-color 0.2s',
                whiteSpace: 'nowrap'
              }}
              onMouseOver={(e) => !loading && (e.currentTarget.style.backgroundColor = '#2563eb')}
              onMouseOut={(e) => !loading && (e.currentTarget.style.backgroundColor = '#3b82f6')}
            >
              {loading ? '⏳ Loading...' : '📊 Analyze'}
            </button>
          </div>
          {selectedCounty && (
            <div style={{ marginTop: '12px', fontSize: '14px', color: '#475569' }}>
              <strong style={{ color: '#2563eb' }}>{selectedCounty}</strong> - {permitCount} drilling permits (last 6 months)
            </div>
          )}
        </form>
      </div>

      {/* Main Content Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px', marginBottom: '20px' }}>
        
        {/* LEFT COLUMN: Formations & Map */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          
          {/* Major Formations */}
          <div style={{
            backgroundColor: 'white',
            padding: '20px',
            borderRadius: '12px',
            boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
          }}>
            <h2 style={{ 
              margin: '0 0 15px 0', 
              color: '#7c3aed', 
              fontSize: '18px',
              fontWeight: '600',
              borderBottom: '2px solid #a78bfa',
              paddingBottom: '8px'
            }}>
              🛢️ Major Formations Drilled
            </h2>
            {!selectedCounty && !loading ? (
              <p style={{ color: '#64748b', fontSize: '13px', fontStyle: 'italic' }}>
                Enter a county to view formation data...
              </p>
            ) : loading ? (
              <p style={{ color: '#64748b', fontSize: '13px' }}>Loading...</p>
            ) : dashboardData?.formations && dashboardData.formations.length > 0 ? (
              <ul style={{ 
                listStyle: 'none', 
                padding: 0, 
                margin: 0,
                fontSize: '13px',
                lineHeight: '1.6'
              }}>
                {dashboardData.formations.slice(0, 5).map((formation: any, idx: number) => (
                  <li key={idx} style={{ 
                    padding: '10px 0',
                    borderBottom: idx < Math.min(4, dashboardData.formations.length - 1) ? '1px solid #e2e8f0' : 'none',
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center'
                  }}>
                    <span>
                      <strong style={{ color: '#334155', fontSize: '13px' }}>{formation.name}</strong>
                    </span>
                    <span style={{ 
                      backgroundColor: '#ede9fe', 
                      color: '#7c3aed', 
                      padding: '3px 10px', 
                      borderRadius: '12px',
                      fontSize: '12px',
                      fontWeight: '600'
                    }}>
                      {formation.count}
                    </span>
                  </li>
                ))}
              </ul>
            ) : (
              <p style={{ color: '#64748b', fontSize: '13px' }}>
                No formation data available
              </p>
            )}
          </div>

          {/* Interactive Map */}
          <div style={{
            backgroundColor: 'white',
            padding: '20px',
            borderRadius: '12px',
            boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
            minHeight: '400px'
          }}>
            <h2 style={{ 
              margin: '0 0 15px 0', 
              color: '#1e40af', 
              fontSize: '18px',
              fontWeight: '600',
              borderBottom: '2px solid #3b82f6',
              paddingBottom: '8px'
            }}>
              📍 Activity Map
            </h2>
            <div style={{ height: '380px', position: 'relative', borderRadius: '8px', overflow: 'hidden' }}>
              <MapConnect />
            </div>
          </div>
        </div>

        {/* RIGHT COLUMN: Lease Offers & Summary */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          
          {/* Lease Offers */}
          <div style={{
            backgroundColor: 'white',
            padding: '20px',
            borderRadius: '12px',
            boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
          }}>
            <h2 style={{ 
              margin: '0 0 15px 0', 
              color: '#059669', 
              fontSize: '18px',
              fontWeight: '600',
              borderBottom: '2px solid #10b981',
              paddingBottom: '8px'
            }}>
              💰 Recent Reported Lease Offers
            </h2>
            {!selectedCounty && !loading ? (
              <p style={{ color: '#64748b', fontSize: '13px', fontStyle: 'italic' }}>
                Enter a county to view reported lease offers from the mineral rights forum...
              </p>
            ) : loading ? (
              <p style={{ color: '#64748b', fontSize: '13px' }}>Loading forum data...</p>
            ) : (
              <div style={{ fontSize: '13px', color: '#475569' }}>
                <div style={{ 
                  padding: '15px',
                  backgroundColor: '#f0fdf4',
                  borderRadius: '8px',
                  border: '1px solid #86efac',
                  marginBottom: '8px'
                }}>
                  <p style={{ margin: '0 0 10px 0', fontSize: '12px', color: '#166534', fontWeight: '600' }}>
                    📊 Forum-Reported Data - {selectedCounty}:
                  </p>
                  <div style={{ 
                    fontSize: '13px', 
                    color: '#15803d', 
                    lineHeight: '1.7',
                    whiteSpace: 'pre-wrap',
                    maxHeight: '150px',
                    overflowY: 'auto'
                  }}>
                    {dashboardData?.leaseOffers || 'Loading...'}
                  </div>
                </div>
                <p style={{ fontSize: '11px', color: '#64748b', fontStyle: 'italic' }}>
                  * Data sourced from mineral rights forum discussions
                </p>
              </div>
            )}
          </div>

          {/* High-Level Summary */}
          <div style={{
            backgroundColor: 'white',
            padding: '20px',
            borderRadius: '12px',
            boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
            flex: 1
          }}>
            <h2 style={{ 
              margin: '0 0 15px 0', 
              color: '#dc2626', 
              fontSize: '18px',
              fontWeight: '600',
              borderBottom: '2px solid #ef4444',
              paddingBottom: '8px'
            }}>
              📈 High-Level Summary
            </h2>
            {!selectedCounty && !loading ? (
              <p style={{ color: '#64748b', fontSize: '13px', fontStyle: 'italic' }}>
                Enter a county to view market analysis...
              </p>
            ) : loading ? (
              <p style={{ color: '#64748b', fontSize: '13px' }}>Generating AI analysis...</p>
            ) : (
              <div style={{ 
                margin: 0, 
                color: '#475569', 
                fontSize: '13px',
                lineHeight: '1.7',
                textAlign: 'justify',
                whiteSpace: 'pre-wrap',
                maxHeight: '400px',
                overflowY: 'auto'
              }}>
                {dashboardData?.marketSummary}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Footer Info */}
      <div style={{
        backgroundColor: '#f8fafc',
        padding: '15px 20px',
        borderRadius: '8px',
        border: '1px solid #e2e8f0',
        fontSize: '12px',
        color: '#64748b',
        textAlign: 'center'
      }}>
        <strong style={{ color: '#334155' }}>ℹ️ Data Sources:</strong> Oklahoma Corporation Commission (OCC), Texas Railroad Commission (RRC). 
        Drilling permits from last 6 months. Market analysis powered by AI.
      </div>
    </div>
  );
};

export default LeasingDashboard;

