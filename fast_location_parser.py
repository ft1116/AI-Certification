#!/usr/bin/env python3
"""
Fast location parsing using off-the-shelf packages
Replaces slow geocoding with instant lookups
"""

from geonamescache import GeonamesCache
import re

class FastLocationParser:
    def __init__(self):
        self.gc = GeonamesCache()
        self.us_states = self.gc.get_us_states()
        self.us_counties = self.gc.get_us_counties()
        
        # Oil & gas states for weighting
        self.oil_gas_states = {
            'texas', 'oklahoma', 'louisiana', 'north dakota', 'colorado', 
            'pennsylvania', 'ohio', 'west virginia', 'wyoming', 'new mexico', 
            'kansas', 'arkansas'
        }
    
    def parse_location(self, query):
        """Fast location parsing with oil & gas weighting"""
        query_lower = query.lower()
        
        # Extract county and state patterns
        county_state_pattern = r'([a-zA-Z\s]+(?:county|city|town)),?\s*([a-zA-Z]{2,})'
        matches = re.findall(county_state_pattern, query_lower)
        
        if matches:
            county_name = matches[0][0].strip()
            state_name = matches[0][1].strip()
            
            # Clean up county name
            county_name = re.sub(r'\s+(county|city|town)$', '', county_name)
            
            # Get coordinates from local database (instant)
            location_data = self._get_coordinates_fast(county_name, state_name)
            
            if location_data:
                # Apply oil & gas weighting
                is_oil_gas = state_name.lower() in self.oil_gas_states
                weight = 2.0 if is_oil_gas else 1.0
                
                print(f"🎯 Fast location: {county_name}, {state_name}")
                print(f"🎯 Oil & gas weighting: {weight}x")
                
                return {
                    "type": "county",
                    "name": f"{county_name}, {state_name}",
                    "coordinates": location_data["coordinates"],
                    "zoom": 9,
                    "weight": weight,
                    "address": location_data.get("address", ""),
                    "confidence": "high"
                }
        
        return None
    
    def _get_coordinates_fast(self, county, state):
        """Get coordinates from local database (instant)"""
        try:
            # Try to find in counties database
            for state_code, counties in self.us_counties.items():
                state_name = self.us_states.get(state_code, {}).get('name', '').lower()
                if state_name == state.lower():
                    for county_code, county_data in counties.items():
                        if county.lower() in county_data['name'].lower():
                            return {
                                "coordinates": [county_data['longitude'], county_data['latitude']],
                                "address": f"{county_data['name']}, {state_name.title()}"
                            }
            
            # Fallback to state center
            state_data = None
            for state_code, data in self.us_states.items():
                if data['name'].lower() == state.lower():
                    state_data = data
                    break
            
            if state_data:
                return {
                    "coordinates": [state_data['longitude'], state_data['latitude']],
                    "address": f"{state.title()}"
                }
                
        except Exception as e:
            print(f"Error in fast geocoding: {e}")
        
        return None

# Test the fast parser
if __name__ == "__main__":
    parser = FastLocationParser()
    
    test_queries = [
        "Dawson County, Texas",
        "Cook County, Illinois", 
        "Grady County, Oklahoma",
        "Washington County, Pennsylvania"
    ]
    
    print("🚀 Testing Fast Location Parser")
    print("=" * 50)
    
    for query in test_queries:
        print(f"\n🧪 Testing: {query}")
        result = parser.parse_location(query)
        if result:
            print(f"✅ Found: {result['name']}")
            print(f"📍 Coordinates: {result['coordinates']}")
            print(f"⚖️ Weight: {result['weight']}x")
        else:
            print("❌ Not found")
