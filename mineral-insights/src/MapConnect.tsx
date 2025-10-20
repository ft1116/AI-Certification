import React, { useEffect, useRef, useState, useCallback } from 'react';
import { Map, View } from 'ol';
import TileLayer from 'ol/layer/Tile';
import OSM from 'ol/source/OSM';
import VectorLayer from 'ol/layer/Vector';
import VectorSource from 'ol/source/Vector';
import GeoJSON from 'ol/format/GeoJSON';
import { Feature } from 'ol';
import { Point } from 'ol/geom';
import { Style, Icon, Circle, Fill, Stroke } from 'ol/style';
import { transform } from 'ol/proj';
import { useLocation } from './LocationContext';
import 'ol/ol.css';

const MapConnect: React.FC = () => {
  const mapRef = useRef<HTMLDivElement>(null);
  const mapInstanceRef = useRef<Map | null>(null);
  const { location } = useLocation();
  const [wellsLoaded, setWellsLoaded] = useState(false);
  const [showWells, setShowWells] = useState(false); // Start with wells hidden
  const [wellsCount, setWellsCount] = useState<number | null>(null);
  const [loadingWells, setLoadingWells] = useState(false);
  const [permits, setPermits] = useState<any[]>([]);
  const [loadingPermits, setLoadingPermits] = useState(false);
  const [showPermits, setShowPermits] = useState(true); // Show permits by default
  const permitLayerRef = useRef<VectorLayer<VectorSource> | null>(null);

  useEffect(() => {
    if (!mapRef.current) {
      console.log('Map container not found');
      return;
    }

    console.log('Creating OpenLayers map...');

    // Create the map with CartoDB tiles that show county boundaries more clearly
    const map = new Map({
      target: mapRef.current,
      layers: [
        new TileLayer({
          source: new OSM({
            url: 'https://{a-c}.tile.openstreetmap.org/{z}/{x}/{y}.png',
            attributions: '© OpenStreetMap contributors',
          }),
        }),
      ],
      view: new View({
        center: transform([-97.5164, 35.4676], 'EPSG:4326', 'EPSG:3857'), // Oklahoma City coordinates transformed
        zoom: 7,
      }),
    });

    console.log('Map created successfully');
    console.log('Map instance stored in ref:', !!mapInstanceRef.current);

    // Create a vector source for mineral rights data
    const vectorSource = new VectorSource();

    // Add sample mineral rights data points
    const mineralRightsData = [
      {
        name: 'Oklahoma City Oil Field',
        coordinates: [-97.5164, 35.4676],
        type: 'Oil & Gas',
        status: 'Active'
      },
      {
        name: 'Tulsa Mining Site',
        coordinates: [-95.9928, 36.1540],
        type: 'Coal',
        status: 'Active'
      },
      {
        name: 'Norman Quarry',
        coordinates: [-97.4395, 35.2226],
        type: 'Limestone',
        status: 'Active'
      }
    ];

    // Create features for each mineral rights location
    mineralRightsData.forEach(location => {
      // Transform coordinates from WGS84 to Web Mercator
      const transformedCoords = transform(location.coordinates, 'EPSG:4326', 'EPSG:3857');
      
      const feature = new Feature({
        geometry: new Point(transformedCoords),
        name: location.name,
        type: location.type,
        status: location.status
      });

      // Create a custom style for mineral rights markers
      feature.setStyle(new Style({
        image: new Icon({
          src: 'data:image/svg+xml;base64,' + btoa(`
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <circle cx="12" cy="12" r="10" fill="#3B82F6" stroke="#1E40AF" stroke-width="2"/>
              <path d="M8 12l2 2 4-4" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
          `),
          scale: 1.5
        })
      }));

      vectorSource.addFeature(feature);
    });

    // Add the vector layer to the map
    const vectorLayer = new VectorLayer({
      source: vectorSource,
    });

    map.addLayer(vectorLayer);

    // Add US county boundaries layer with bold styling
    const countySource = new VectorSource({
      url: 'https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json',
      format: new GeoJSON(),
    });

    const countyLayer = new VectorLayer({
      source: countySource,
      style: new Style({
        stroke: new Stroke({
          color: 'rgba(100, 100, 100, 0.4)', // Light gray with transparency
          width: 0.8, // Very thin line width
        }),
        fill: new Fill({
          color: 'rgba(0, 0, 0, 0)', // Transparent fill
        }),
      }),
      zIndex: 1, // Render just above base map, behind all markers and data layers
    });

    map.addLayer(countyLayer);

    // Store map instance for cleanup
    mapInstanceRef.current = map;

    // Cleanup function
    return () => {
      if (mapInstanceRef.current) {
        mapInstanceRef.current.setTarget(undefined);
      }
    };
  }, []);

  // Function to check wells count
  const checkWellsCount = async () => {
    try {
      const response = await fetch('http://localhost:8003/wells/count');
      const data = await response.json();
      setWellsCount(data.count);
      console.log('Total wells in database:', data.count);
    } catch (error) {
      console.error('Error checking wells count:', error);
    }
  };

  // Function to load wells data with performance warning
  const loadWellsData = useCallback(async () => {
    if (wellsLoaded || !mapInstanceRef.current) return;
    
    // Check if there are too many wells
    if (wellsCount && wellsCount > 100000) {
      const proceed = window.confirm(
        `Warning: There are ${wellsCount.toLocaleString()} wells in the database. ` +
        `Loading all wells may cause performance issues or freeze your browser. ` +
        `Would you like to proceed? (Recommended: Use zoom-based loading instead)`
      );
      
      if (!proceed) {
        setShowWells(false);
        return;
      }
    }
    
    setLoadingWells(true);
    
    try {
      console.log('Loading wells data...');
      const response = await fetch('http://localhost:8003/wells');
      const wellsData = await response.json();
      
      console.log('Wells data loaded:', wellsData.features?.length, 'wells');
      
      // Create wells vector source
      const wellsSource = new VectorSource({
        features: new GeoJSON().readFeatures(wellsData, {
          featureProjection: 'EPSG:3857'
        })
      });
      
      // Create wells vector layer with styling
      const wellsLayer = new VectorLayer({
        source: wellsSource,
        style: (feature) => {
          const properties = feature.getProperties();
          const wellStatus = properties.wellstatus;
          const wellType = properties.welltype;
          
          // Color coding based on well status and type
          let color = '#666666'; // Default gray
          
          if (wellStatus === 'PA') {
            color = '#dc2626'; // Red for plugged/abandoned
          } else if (wellStatus === 'AC' || wellStatus === 'ACTIVE') {
            color = '#16a34a'; // Green for active
          } else if (wellType === 'OIL') {
            color = '#1e40af'; // Blue for oil wells
          } else if (wellType === 'GAS') {
            color = '#7c3aed'; // Purple for gas wells
          } else if (wellType === 'DRY') {
            color = '#dc2626'; // Red for dry holes
          }
          
          return new Style({
            image: new Circle({
              radius: 3,
              fill: new Fill({ color: color }),
              stroke: new Stroke({ color: '#ffffff', width: 1 })
            })
          });
        }
      });
      
      // Add wells layer to map
      mapInstanceRef.current.addLayer(wellsLayer);
      setWellsLoaded(true);
      console.log('Wells layer added to map');
      
    } catch (error) {
      console.error('Error loading wells data:', error);
      alert('Error loading wells data. The file may be too large for your browser to handle.');
    } finally {
      setLoadingWells(false);
    }
  }, [wellsLoaded, wellsCount]);

  // Check wells count on component mount
  useEffect(() => {
    checkWellsCount();
  }, []);

  // Load wells data when map is ready
  useEffect(() => {
    if (mapInstanceRef.current && showWells && !wellsLoaded) {
      loadWellsData();
    }
  }, [showWells, wellsLoaded, loadWellsData]);

  // Handle location changes
  useEffect(() => {
    console.log('🗺️ MapConnect location effect triggered:', location);
    console.log('🗺️ Map instance available:', !!mapInstanceRef.current);
    if (!mapInstanceRef.current || !location) {
      console.log('🗺️ Map or location not available:', { map: !!mapInstanceRef.current, location });
      return;
    }

    const map = mapInstanceRef.current;
    const view = map.getView();
    console.log('Animating to location:', location);

    if (location.coordinates) {
      // Transform coordinates from WGS84 (EPSG:4326) to Web Mercator (EPSG:3857)
      const transformedCoords = transform(location.coordinates, 'EPSG:4326', 'EPSG:3857');
      
      // Zoom to coordinates (for cities, counties, states)
      console.log('Original coordinates:', location.coordinates);
      console.log('Transformed coordinates:', transformedCoords);
      console.log('Animating to coordinates with zoom:', location.zoom || 10);
      
      view.animate({
        center: transformedCoords,
        zoom: location.zoom || 10,
        duration: 1000
      });
      console.log('Animation started for coordinates');
    } else if (location.type === 'str' && location.coordinates) {
      // Transform STR coordinates from WGS84 (EPSG:4326) to Web Mercator (EPSG:3857)
      const transformedCoords = transform(location.coordinates, 'EPSG:4326', 'EPSG:3857');
      
      // Use coordinates from backend (converted using proper geospatial calculations)
      console.log('Original STR coordinates:', location.coordinates);
      console.log('Transformed STR coordinates:', transformedCoords);
      console.log('Animating to STR coordinates with zoom:', location.zoom || 12);
      
      view.animate({
        center: transformedCoords,
        zoom: location.zoom || 12,
        duration: 1000
      });
      console.log('Animation started for STR coordinates');
    }
    
    // Load permits for the new location
    if (location && (location.type === 'county' || location.type === 'city' || location.type === 'str')) {
      loadPermitsForLocation(location);
    }
    
    // Handle direct permit data from chat
    if (location && location.type === 'permits' && location.data) {
      console.log('🗺️ Received permit data from chat:', location.data);
      console.log('🗺️ GeoJSON features count:', location.data.features?.length);
      console.log('🗺️ First feature sample:', location.data.features?.[0]);
      displayPermitData(location.data);
    }
  }, [location]);

  // Function to display permit data from chat
  const displayPermitData = useCallback((geojsonData: any) => {
    console.log('🔍 displayPermitData called with:', geojsonData);
    console.log('🔍 Map instance exists:', !!mapInstanceRef.current);
    console.log('🔍 GeoJSON data exists:', !!geojsonData);
    console.log('🔍 Features exist:', !!geojsonData?.features);
    console.log('🔍 Features count:', geojsonData?.features?.length);
    
    if (!mapInstanceRef.current || !geojsonData || !geojsonData.features) {
      console.log('❌ No map instance or permit data available');
      return;
    }

    console.log(`✅ Displaying ${geojsonData.features.length} permits on map`);

    // Remove existing permit layer
    if (permitLayerRef.current) {
      mapInstanceRef.current.removeLayer(permitLayerRef.current);
    }

    // Create new vector source from GeoJSON with proper projection
    const features = new GeoJSON().readFeatures(geojsonData, {
      dataProjection: 'EPSG:4326',
      featureProjection: 'EPSG:3857'
    });
    
    console.log('Created features:', features.length);
    if (features[0]) {
      const geometry = features[0].getGeometry();
      if (geometry && 'getCoordinates' in geometry) {
        console.log('First feature geometry:', (geometry as any).getCoordinates());
      }
    }
    
    const vectorSource = new VectorSource({
      features: features
    });

    // Create permit layer with red dots
    const permitLayer = new VectorLayer({
      source: vectorSource,
      style: new Style({
        image: new Circle({
          radius: 12, // Make dots bigger
          fill: new Fill({ color: '#ff0000' }), // Bright red color
          stroke: new Stroke({ color: '#ffffff', width: 3 }) // Thicker white border
        })
      })
    });

    // Set high z-index to ensure dots appear above other layers
    permitLayer.setZIndex(1000);

    // Add layer to map
    console.log('🔍 Adding permit layer to map...');
    mapInstanceRef.current.addLayer(permitLayer);
    permitLayerRef.current = permitLayer;
    console.log('✅ Permit layer added to map');
    
    // Debug: Check if layer is actually on the map
    const layers = mapInstanceRef.current.getLayers();
    console.log('🔍 Total layers on map:', layers.getLength());
    console.log('🔍 Permit layer visible:', permitLayer.getVisible());
    console.log('🔍 Permit layer z-index:', permitLayer.getZIndex());

    // Fit map to show all permits
    const extent = vectorSource.getExtent();
    console.log('🔍 Permit extent:', extent);
    if (extent && extent[0] !== Infinity) {
      console.log('🔍 Fitting map to permit extent...');
      mapInstanceRef.current.getView().fit(extent, {
        padding: [50, 50, 50, 50],
        duration: 1000
      });
      console.log('✅ Map fitted to permit extent');
    } else {
      console.log('❌ No valid extent found for permits');
    }

    setPermits(geojsonData.features);
    setShowPermits(true);
    
    // Test: Add a simple test dot to see if rendering works
    console.log('🔍 Adding test dot...');
    const testFeature = new Feature({
      geometry: new Point([-97.99682509, 35.07118486]) // Same coordinates as first permit
    });
    const testSource = new VectorSource({
      features: [testFeature]
    });
    const testLayer = new VectorLayer({
      source: testSource,
      style: new Style({
        image: new Circle({
          radius: 20, // Very big test dot
          fill: new Fill({ color: '#00ff00' }), // Bright green
          stroke: new Stroke({ color: '#000000', width: 5 })
        })
      })
    });
    testLayer.setZIndex(2000); // Even higher z-index
    mapInstanceRef.current.addLayer(testLayer);
    console.log('✅ Test dot added');
  }, []);

  // Function to load permits for a location
  const loadPermitsForLocation = useCallback(async (locationData: any) => {
    if (!locationData) return;
    
    setLoadingPermits(true);
    try {
      let url = 'http://localhost:8003/permits/location?';
      
      // Build query params based on location type
      if (locationData.type === 'str' && locationData.section && locationData.township && locationData.range) {
        // Section-Township-Range query
        url += `section=${encodeURIComponent(locationData.section)}`;
        url += `&township=${encodeURIComponent(locationData.township)}`;
        url += `&range_str=${encodeURIComponent(locationData.range)}`;
        url += '&radius_miles=10'; // Smaller radius for STR queries
      } else if (locationData.type === 'county' && locationData.name) {
        // Extract county name from full location string (e.g., "karnes county, texas" -> "karnes")
        const countyMatch = locationData.name.match(/([a-z\s]+)\s+county/i);
        const countyName = countyMatch ? countyMatch[1].trim() : locationData.name.split(',')[0].trim();
        url += `county=${encodeURIComponent(countyName)}`;
      } else if (locationData.coordinates) {
        url += `lat=${locationData.coordinates[1]}&lng=${locationData.coordinates[0]}`;
      }
      
      url += '&limit=100';
      
      console.log('Fetching permits from:', url);
      const response = await fetch(url);
      const data = await response.json();
      
      console.log('Permits data received:', data);
      setPermits(data.permits || []);
      
      // Display permits on map
      if (data.permits && data.permits.length > 0) {
        displayPermitsOnMap(data.permits);
      }
    } catch (error) {
      console.error('Error loading permits:', error);
    } finally {
      setLoadingPermits(false);
    }
  }, []);

  // Function to display permits as markers on the map
  const displayPermitsOnMap = useCallback((permitsData: any[]) => {
    if (!mapInstanceRef.current) return;
    
    const map = mapInstanceRef.current;
    
    // Remove existing permit layer if any
    if (permitLayerRef.current) {
      map.removeLayer(permitLayerRef.current);
    }
    
    // Create features for permits with coordinates
    const features = permitsData
      .filter(permit => permit.latitude && permit.longitude)
      .map(permit => {
        const feature = new Feature({
          geometry: new Point(transform(
            [permit.longitude, permit.latitude],
            'EPSG:4326',
            'EPSG:3857'
          )),
          permit: permit
        });
        
        return feature;
      });
    
    console.log(`Created ${features.length} permit markers`);
    
    // Create vector source and layer
    const permitSource = new VectorSource({
      features: features
    });
    
    const permitLayer = new VectorLayer({
      source: permitSource,
      style: new Style({
        image: new Circle({
          radius: 6,
          fill: new Fill({ color: 'rgba(255, 99, 71, 0.8)' }),
          stroke: new Stroke({ color: '#fff', width: 2 })
        })
      })
    });
    
    permitLayerRef.current = permitLayer;
    map.addLayer(permitLayer);
    
    // Add click handler for permit markers
    map.on('click', (evt) => {
      const feature = map.forEachFeatureAtPixel(evt.pixel, (f) => f);
      if (feature && feature.get('permit')) {
        const permit = feature.get('permit');
        alert(`${permit.lease_name}\nOperator: ${permit.operator}\nAPI: ${permit.api_number}\nCounty: ${permit.county}`);
      }
    });
  }, []);

  return (
    <div style={{ textAlign: 'center' }}>
      {/* Map Container */}
      <div style={{ 
        width: '100%', 
        height: '500px', 
        border: '2px solid rgba(16, 185, 129, 0.3)', 
        borderRadius: '16px', 
        overflow: 'hidden', 
        backgroundColor: 'rgba(30, 41, 59, 0.4)',
        boxShadow: '0 4px 6px rgba(0,0,0,0.2)',
        marginBottom: '20px'
      }}>
        <div 
          ref={mapRef} 
          style={{ width: '100%', height: '100%', minHeight: '500px' }}
        />
      </div>

      {/* Permits Status */}
      {loadingPermits && (
        <div style={{ 
          marginBottom: '16px', 
          padding: '14px', 
          backgroundColor: 'rgba(59, 130, 246, 0.15)', 
          border: '1px solid rgba(59, 130, 246, 0.3)', 
          borderRadius: '12px',
          color: '#3b82f6',
          fontSize: '13px'
        }}>
          Loading drilling permits...
        </div>
      )}
      
      {permits.length > 0 && (
        <div style={{ 
          marginBottom: '16px', 
          padding: '14px', 
          backgroundColor: 'rgba(16, 185, 129, 0.15)', 
          border: '1px solid rgba(16, 185, 129, 0.3)', 
          borderRadius: '12px',
          color: '#10b981',
          fontSize: '13px'
        }}>
          <strong>📍 {permits.length} drilling permits found</strong> - {permits.filter(p => p.latitude && p.longitude).length} shown on map (last 6 months)
        </div>
      )}

      {/* Wells Control */}
      {wellsCount && wellsCount > 150000 && (
        <div style={{ 
          marginBottom: '16px', 
          padding: '14px', 
          backgroundColor: 'rgba(251, 191, 36, 0.15)', 
          border: '1px solid rgba(251, 191, 36, 0.3)', 
          borderRadius: '12px',
          color: '#fbbf24',
          fontSize: '13px',
          textAlign: 'left'
        }}>
          <strong>⚠️ Performance Warning:</strong> The wells database contains {wellsCount.toLocaleString()} wells. 
          Loading all wells may cause performance issues.
        </div>
      )}
      {wellsCount && wellsCount <= 150000 && wellsCount > 50000 && (
        <div style={{ 
          marginBottom: '16px', 
          padding: '14px', 
          backgroundColor: 'rgba(16, 185, 129, 0.15)', 
          border: '1px solid rgba(16, 185, 129, 0.3)', 
          borderRadius: '12px',
          color: '#10b981',
          fontSize: '13px',
          textAlign: 'left'
        }}>
          <strong>✅ Optimized Dataset:</strong> Showing {wellsCount.toLocaleString()} active and operational wells 
          (plugged and abandoned wells filtered).
        </div>
      )}
      
      <div style={{ marginBottom: '16px', display: 'flex', gap: '10px', justifyContent: 'center' }}>
        <button 
          onClick={() => setShowWells(!showWells)}
          disabled={loadingWells}
          style={{ 
            padding: '12px 20px', 
            backgroundColor: showWells ? '#dc2626' : '#10b981', 
            color: 'white', 
            borderRadius: '10px', 
            border: 'none',
            cursor: loadingWells ? 'not-allowed' : 'pointer',
            fontSize: '14px',
            fontWeight: '600',
            opacity: loadingWells ? 0.6 : 1,
            boxShadow: '0 2px 4px rgba(0,0,0,0.2)'
          }}
        >
          {loadingWells ? '⏳ Loading...' : 
           showWells ? '👁️ Hide Wells' : 
           wellsCount ? `🔍 Show Wells (${wellsCount.toLocaleString()})` : '🔍 Show Wells'}
        </button>
      </div>

      {/* Legend */}
      <div style={{ 
        padding: '16px',
        backgroundColor: 'rgba(15, 23, 42, 0.5)',
        borderRadius: '12px',
        border: '1px solid rgba(16, 185, 129, 0.2)',
        fontSize: '13px',
        color: 'rgba(255, 255, 255, 0.8)',
        textAlign: 'left'
      }}>
        <p style={{ margin: '0 0 8px 0', fontWeight: '600', color: 'white' }}>Map Legend:</p>
        <p style={{ margin: '4px 0' }}>• Blue markers: Active mineral rights locations</p>
        {showWells && (
          <>
            <p style={{ margin: '4px 0' }}>• <span style={{color: '#16a34a', fontWeight: 'bold'}}>●</span> Green: Active wells</p>
            <p style={{ margin: '4px 0' }}>• <span style={{color: '#1e40af', fontWeight: 'bold'}}>●</span> Blue: Oil wells</p>
            <p style={{ margin: '4px 0' }}>• <span style={{color: '#7c3aed', fontWeight: 'bold'}}>●</span> Purple: Gas wells</p>
            <p style={{ margin: '4px 0' }}>• <span style={{color: '#dc2626', fontWeight: 'bold'}}>●</span> Red: Plugged/Dry wells</p>
          </>
        )}
        <p style={{ margin: '8px 0 0 0', fontSize: '12px', color: 'rgba(255, 255, 255, 0.6)' }}>
          💡 Tip: Ask about specific cities or locations in the chat to zoom to them!
        </p>
      </div>
    </div>
  );
};

export default MapConnect;