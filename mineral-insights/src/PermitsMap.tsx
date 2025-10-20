import React, { useState, useEffect } from 'react';
import { MapContainer, TileLayer, Marker, Popup } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';

// Fix for default markers in Leaflet with React
delete (L.Icon.Default.prototype as any)._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
});

interface PermitFeature {
  type: 'Feature';
  properties: {
    api_number: string;
    operator: string;
    well_name: string;
    well_type: string;
    county: string;
    section?: string;
    township?: string;
    range?: string;
    latitude: number;
    longitude: number;
    formation_name?: string;
    total_depth?: number;
    approval_date: string;
    permit_status: string;
    remarks?: string;
  };
  geometry: {
    type: 'Point';
    coordinates: [number, number];
  };
}

interface PermitsMapProps {
  query: string;
}

const PermitsMap: React.FC<PermitsMapProps> = ({ query }) => {
  const [permitsData, setPermitsData] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [mapCenter, setMapCenter] = useState<[number, number]>([35.5, -97.5]); // Oklahoma center

  // Fetch permits data when query changes
  useEffect(() => {
    if (query) {
      fetchPermitsData(query);
    }
  }, [query]);

  const fetchPermitsData = async (searchQuery: string) => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch('http://localhost:8003/map/permits', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: searchQuery,
          conversation_id: `map_${Date.now()}`
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setPermitsData(data);

      // Update map center if we have coordinates
      if (data.summary && data.summary.has_coordinates > 0) {
        const centerLat = (data.bounds.north + data.bounds.south) / 2;
        const centerLng = (data.bounds.east + data.bounds.west) / 2;
        setMapCenter([centerLat, centerLng]);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch permits data');
      console.error('Error fetching permits:', err);
    } finally {
      setLoading(false);
    }
  };

  const getMarkerColor = (wellType: string, operator: string) => {
    const type = wellType.toLowerCase();
    const op = operator.toLowerCase();
    
    // Color by well type
    if (type.includes('oil')) return '#e74c3c'; // Red
    if (type.includes('gas')) return '#3498db'; // Blue
    if (type.includes('horizontal')) return '#9b59b6'; // Purple
    if (type.includes('og')) return '#f39c12'; // Orange for OG (Oil & Gas)
    
    // Default color
    return '#95a5a6'; // Gray
  };

  const createCustomIcon = (color: string) => {
    return L.divIcon({
      className: 'custom-div-icon',
      html: `<div style="
        background-color: ${color};
        width: 12px;
        height: 12px;
        border-radius: 50%;
        border: 2px solid white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.3);
      "></div>`,
      iconSize: [16, 16],
      iconAnchor: [8, 8]
    });
  };

  if (loading) {
    return (
      <div style={{ 
        height: '500px', 
        display: 'flex', 
        alignItems: 'center', 
        justifyContent: 'center',
        backgroundColor: '#f8f9fa',
        borderRadius: '8px',
        border: '1px solid #dee2e6'
      }}>
        <div style={{ textAlign: 'center' }}>
          <div style={{ fontSize: '24px', marginBottom: '10px' }}>🗺️</div>
          <div>Loading permits map...</div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div style={{ 
        height: '500px', 
        display: 'flex', 
        alignItems: 'center', 
        justifyContent: 'center',
        backgroundColor: '#f8d7da',
        color: '#721c24',
        borderRadius: '8px',
        border: '1px solid #f5c6cb'
      }}>
        <div style={{ textAlign: 'center' }}>
          <div style={{ fontSize: '24px', marginBottom: '10px' }}>❌</div>
          <div>Error: {error}</div>
        </div>
      </div>
    );
  }

  if (!permitsData || permitsData.summary.total_permits === 0) {
    return (
      <div style={{ 
        height: '500px', 
        display: 'flex', 
        alignItems: 'center', 
        justifyContent: 'center',
        backgroundColor: '#f8f9fa',
        borderRadius: '8px',
        border: '1px solid #dee2e6'
      }}>
        <div style={{ textAlign: 'center' }}>
          <div style={{ fontSize: '24px', marginBottom: '10px' }}>🔍</div>
          <div>No permits found for: "{query}"</div>
          <div style={{ fontSize: '14px', color: '#6c757d', marginTop: '10px' }}>
            Try different search terms or check your spelling.
          </div>
        </div>
      </div>
    );
  }

  return (
    <div style={{ position: 'relative' }}>
      {/* Summary Stats */}
      <div style={{ 
        backgroundColor: 'white', 
        padding: '15px', 
        borderRadius: '8px',
        marginBottom: '15px',
        boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
      }}>
        <h4 style={{ margin: '0 0 10px 0', color: '#2c3e50' }}>
          📊 Search Results: "{query}"
        </h4>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '15px' }}>
          <div>
            <strong style={{ color: '#7f8c8d' }}>Total Permits:</strong>
            <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#3498db' }}>
              {permitsData.summary.total_permits}
            </div>
          </div>
          <div>
            <strong style={{ color: '#7f8c8d' }}>With Coordinates:</strong>
            <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#27ae60' }}>
              {permitsData.summary.has_coordinates}
            </div>
          </div>
          {Object.keys(permitsData.summary.top_operators).length > 0 && (
            <div>
              <strong style={{ color: '#7f8c8d' }}>Top Operator:</strong>
              <div style={{ fontSize: '14px', fontWeight: 'bold', color: '#e67e22' }}>
                {Object.keys(permitsData.summary.top_operators)[0]} ({permitsData.summary.top_operators[Object.keys(permitsData.summary.top_operators)[0]]})
              </div>
            </div>
          )}
        </div>
        
        {Object.keys(permitsData.summary.top_operators).length > 0 && (
          <div style={{ marginTop: '10px' }}>
            <strong style={{ fontSize: '12px', color: '#7f8c8d' }}>Top Operators:</strong>
            <ul style={{ margin: '5px 0', paddingLeft: '15px', fontSize: '12px' }}>
              {Object.entries(permitsData.summary.top_operators).slice(0, 3).map(([operator, count]) => (
                <li key={operator}>{operator} ({count as number})</li>
              ))}
            </ul>
          </div>
        )}
      </div>

      {/* Map */}
      <div style={{ height: '500px', width: '100%', borderRadius: '8px', overflow: 'hidden' }}>
        <MapContainer
          center={mapCenter}
          zoom={10}
          style={{ height: '100%', width: '100%' }}
          key={`${mapCenter[0]}-${mapCenter[1]}-${query}`}
        >
          <TileLayer
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          />
          
          {permitsData.geojson.features.map((feature: PermitFeature, index: number) => {
            const { latitude, longitude } = feature.properties;
            if (!latitude || !longitude) return null;

            const color = getMarkerColor(feature.properties.well_type, feature.properties.operator);
            const customIcon = createCustomIcon(color);

            return (
              <Marker
                key={index}
                position={[latitude, longitude]}
                icon={customIcon}
              >
                <Popup>
                  <div style={{ minWidth: '200px' }}>
                    <h4 style={{ margin: '0 0 8px 0', color: '#2c3e50' }}>
                      {feature.properties.well_name}
                    </h4>
                    <p style={{ margin: '4px 0', fontSize: '14px' }}>
                      <strong>Operator:</strong> {feature.properties.operator}
                    </p>
                    <p style={{ margin: '4px 0', fontSize: '14px' }}>
                      <strong>Type:</strong> {feature.properties.well_type}
                    </p>
                    <p style={{ margin: '4px 0', fontSize: '14px' }}>
                      <strong>Location:</strong> {feature.properties.county} County
                    </p>
                    {feature.properties.section && (
                      <p style={{ margin: '4px 0', fontSize: '14px' }}>
                        <strong>Section:</strong> {feature.properties.section}
                        {feature.properties.township && `, ${feature.properties.township}`}
                        {feature.properties.range && `, ${feature.properties.range}`}
                      </p>
                    )}
                    <p style={{ margin: '4px 0', fontSize: '14px' }}>
                      <strong>Status:</strong> {feature.properties.permit_status}
                    </p>
                    <p style={{ margin: '4px 0', fontSize: '14px' }}>
                      <strong>Approved:</strong> {feature.properties.approval_date}
                    </p>
                    {feature.properties.formation_name && (
                      <p style={{ margin: '4px 0', fontSize: '14px' }}>
                        <strong>Formation:</strong> {feature.properties.formation_name}
                      </p>
                    )}
                  </div>
                </Popup>
              </Marker>
            );
          })}
        </MapContainer>
        
        {/* Legend */}
        <div style={{ 
          position: 'absolute', 
          bottom: '10px', 
          right: '10px', 
          backgroundColor: 'white', 
          padding: '10px', 
          borderRadius: '4px',
          boxShadow: '0 2px 4px rgba(0,0,0,0.2)',
          fontSize: '12px'
        }}>
          <div style={{ fontWeight: 'bold', marginBottom: '5px' }}>Legend:</div>
          <div style={{ display: 'flex', alignItems: 'center', marginBottom: '2px' }}>
            <div style={{ width: '12px', height: '12px', backgroundColor: '#e74c3c', borderRadius: '50%', marginRight: '5px' }}></div>
            Oil Wells
          </div>
          <div style={{ display: 'flex', alignItems: 'center', marginBottom: '2px' }}>
            <div style={{ width: '12px', height: '12px', backgroundColor: '#3498db', borderRadius: '50%', marginRight: '5px' }}></div>
            Gas Wells
          </div>
          <div style={{ display: 'flex', alignItems: 'center', marginBottom: '2px' }}>
            <div style={{ width: '12px', height: '12px', backgroundColor: '#9b59b6', borderRadius: '50%', marginRight: '5px' }}></div>
            Horizontal Wells
          </div>
          <div style={{ display: 'flex', alignItems: 'center', marginBottom: '2px' }}>
            <div style={{ width: '12px', height: '12px', backgroundColor: '#f39c12', borderRadius: '50%', marginRight: '5px' }}></div>
            OG Wells
          </div>
          <div style={{ display: 'flex', alignItems: 'center' }}>
            <div style={{ width: '12px', height: '12px', backgroundColor: '#95a5a6', borderRadius: '50%', marginRight: '5px' }}></div>
            Other
          </div>
        </div>
      </div>
    </div>
  );
};

export default PermitsMap;