#!/usr/bin/env python3
"""
Mapping Agent for Oklahoma Drilling Permits
Takes user location queries and returns GeoJSON data for interactive maps
"""

import sqlite3
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re
from datetime import datetime

class DrillingPermitsMapper:
    def __init__(self, db_path: str = "Drilling Permits/data/permits.db"):
        self.db_path = db_path
        
    def parse_location_query(self, query: str) -> Dict:
        """Parse user location query to extract location parameters"""
        query_lower = query.lower()
        
        # Extract county - improved regex to handle "Grady County" pattern
        county_patterns = [
            r'(\w+)\s+county',  # "Grady County"
            r'county\s+(\w+)',  # "county Grady"
            r'in\s+(\w+)\s+county',  # "in Grady County"
        ]
        county = None
        for pattern in county_patterns:
            county_match = re.search(pattern, query_lower)
            if county_match:
                county = county_match.group(1).title()
                break
        
        # Extract state - look for state abbreviations and names
        state_patterns = [
            r'(\w+)\s+county\s+(tx|texas|ok|oklahoma|nm|new mexico)',  # "Leon County TX"
            r'(tx|texas|ok|oklahoma|nm|new mexico)',  # Just state
            r'county\s+(\w+)\s+(tx|texas|ok|oklahoma|nm|new mexico)',  # "county Leon TX"
        ]
        state = None
        for pattern in state_patterns:
            state_match = re.search(pattern, query_lower)
            if state_match:
                state_abbr = state_match.group(-1).lower()  # Get last group (state)
                if state_abbr in ['tx', 'texas']:
                    state = 'Texas'
                elif state_abbr in ['ok', 'oklahoma']:
                    state = 'OK'
                elif state_abbr in ['nm', 'new mexico']:
                    state = 'NM'
                break
        
        # Extract section, township, range (e.g., "section 15 township 15n range 24w")
        section_match = re.search(r'section\s+(\d+)', query_lower)
        section = int(section_match.group(1)) if section_match else None
        
        township_match = re.search(r'township\s+(\d+[ns])', query_lower)
        township = township_match.group(1).upper() if township_match else None
        
        range_match = re.search(r'range\s+(\d+[ew])', query_lower)
        range_val = range_match.group(1).upper() if range_match else None
        
        # Extract operator - improved patterns
        # First check if the entire query looks like an operator name
        if any(phrase in query_lower for phrase in ['oil company', 'energy', 'resources', 'operating', 'llc', 'inc']):
            operator = query.strip().upper()
        else:
            operator_patterns = [
                r'by\s+([^,\n]+)',  # "by MEWBOURNE OIL COMPANY"
                r'operator\s+([^,\n]+)',  # "operator MEWBOURNE"
                r'(\w+\s+oil\s+company)',  # "MEWBOURNE OIL COMPANY"
            ]
            operator = None
            for pattern in operator_patterns:
                operator_match = re.search(pattern, query_lower)
                if operator_match:
                    operator = operator_match.group(1).strip().upper()
                    break
        
        # Extract formation
        formation_match = re.search(r'formation\s+([^,\n]+)', query_lower)
        formation = formation_match.group(1).strip() if formation_match else None
        
        # Extract well type - but not if it's part of a company name
        well_type = None
        # Only extract well type if it's not part of a company name
        if not any(phrase in query_lower for phrase in ['oil company', 'energy', 'resources', 'operating', 'llc', 'inc']):
            well_type_match = re.search(r'(oil|gas|horizontal|vertical)', query_lower)
            well_type = well_type_match.group(1).lower() if well_type_match else None
        
        return {
            'county': county,
            'state': state,
            'section': section,
            'township': township,
            'range': range_val,
            'operator': operator,
            'formation': formation,
            'well_type': well_type,
            'original_query': query
        }
    
    def query_permits(self, location_params: Dict) -> List[Dict]:
        """Query permits database based on location parameters"""
        if not Path(self.db_path).exists():
            return []
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Build dynamic WHERE clause
        conditions = []
        params = []
        
        if location_params['county']:
            conditions.append("LOWER(county) LIKE ?")
            params.append(f"%{location_params['county'].lower()}%")
        
        if location_params['section']:
            conditions.append("section = ?")
            params.append(location_params['section'])
        
        if location_params['township']:
            conditions.append("LOWER(township) LIKE ?")
            params.append(f"%{location_params['township'].lower()}%")
        
        if location_params['range']:
            conditions.append("LOWER(range_val) LIKE ?")
            params.append(f"%{location_params['range'].lower()}%")
        
        if location_params['operator']:
            conditions.append("LOWER(entity_name) LIKE ?")
            params.append(f"%{location_params['operator'].lower()}%")
        
        if location_params['formation']:
            conditions.append("LOWER(formation_name) LIKE ?")
            params.append(f"%{location_params['formation'].lower()}%")
        
        if location_params['well_type']:
            if location_params['well_type'] in ['oil', 'gas']:
                conditions.append("LOWER(well_type) LIKE ?")
                params.append(f"%{location_params['well_type']}%")
            elif location_params['well_type'] == 'horizontal':
                conditions.append("LOWER(well_type) LIKE ?")
                params.append("%horizontal%")
        
        # Add state filtering if available - but be smart about it
        if location_params.get('state'):
            # Only filter by state if it was explicitly mentioned in the query
            conditions.append("LOWER(state) LIKE ?")
            params.append(f"%{location_params['state'].lower()}%")
        else:
            # If no state specified, search all states
            # Since all locations are in the US, this is safe
            # The system will find permits regardless of state
            pass
        
        # If no specific conditions, return all permits
        if not conditions:
            conditions.append("1=1")
        
        sql = f"""
            SELECT api_number, entity_name, well_name, well_type, county, 
                   section, township, range_val, surf_lat_y, surf_long_x, 
                   formation_name, total_depth, NULL as approval_date, well_status,
                   NULL as remarks
            FROM permits 
            WHERE {' AND '.join(conditions)}
            ORDER BY api_number DESC
            LIMIT 100
        """
        
        cursor.execute(sql, params)
        results = cursor.fetchall()
        
        permits = []
        for row in results:
            permits.append({
                'api_number': row[0],
                'operator': row[1],
                'well_name': row[2],
                'well_type': row[3],
                'county': row[4],
                'section': row[5],
                'township': row[6],
                'range': row[7],
                'latitude': row[8],
                'longitude': row[9],
                'formation_name': row[10],
                'total_depth': row[11],
                'approval_date': row[12],
                'permit_status': row[13],
                'remarks': row[14]
            })
        
        conn.close()
        return permits
    
    def create_geojson(self, permits: List[Dict]) -> Dict:
        """Convert permits data to GeoJSON format for mapping"""
        features = []
        
        for permit in permits:
            # Skip permits without coordinates
            if not permit['latitude'] or not permit['longitude']:
                continue
            
            try:
                lat = float(permit['latitude'])
                lon = float(permit['longitude'])
                
                # Skip invalid coordinates
                if lat == 0 and lon == 0:
                    continue
                
                feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [lon, lat]
                    },
                    "properties": {
                        "api_number": permit['api_number'],
                        "operator": permit['operator'],
                        "well_name": permit['well_name'],
                        "well_type": permit['well_type'],
                        "county": permit['county'],
                        "section": permit['section'],
                        "township": permit['township'],
                        "range": permit['range'],
                        "formation": permit['formation_name'],
                        "total_depth": permit['total_depth'],
                        "approval_date": permit['approval_date'],
                        "permit_status": permit['permit_status'],
                        "remarks": permit['remarks'],
                        "popup_content": self._create_popup_content(permit)
                    }
                }
                features.append(feature)
                
            except (ValueError, TypeError):
                continue
        
        geojson = {
            "type": "FeatureCollection",
            "features": features,
            "metadata": {
                "total_permits": len(permits),
                "mapped_permits": len(features),
                "generated_at": datetime.now().isoformat()
            }
        }
        
        return geojson
    
    def _create_popup_content(self, permit: Dict) -> str:
        """Create HTML popup content for map markers"""
        return f"""
        <div style="font-family: Arial, sans-serif; max-width: 300px;">
            <h3 style="margin: 0 0 10px 0; color: #2c3e50;">{permit['well_name']}</h3>
            <p style="margin: 5px 0;"><strong>Operator:</strong> {permit['operator']}</p>
            <p style="margin: 5px 0;"><strong>API:</strong> {permit['api_number']}</p>
            <p style="margin: 5px 0;"><strong>Type:</strong> {permit['well_type']}</p>
            <p style="margin: 5px 0;"><strong>County:</strong> {permit['county']}</p>
            <p style="margin: 5px 0;"><strong>Location:</strong> Section {permit['section']}, {permit['township']}, {permit['range']}</p>
            {f"<p style='margin: 5px 0;'><strong>Formation:</strong> {permit['formation_name']}</p>" if permit['formation_name'] else ""}
            {f"<p style='margin: 5px 0;'><strong>Depth:</strong> {permit['total_depth']} ft</p>" if permit['total_depth'] else ""}
            <p style="margin: 5px 0;"><strong>Approved:</strong> {permit['approval_date']}</p>
            <p style="margin: 5px 0;"><strong>Status:</strong> {permit['permit_status']}</p>
        </div>
        """
    
    def get_map_bounds(self, permits: List[Dict]) -> Optional[Dict]:
        """Calculate map bounds based on permit locations"""
        valid_coords = []
        
        for permit in permits:
            try:
                lat = float(permit['latitude'])
                lon = float(permit['longitude'])
                if lat != 0 and lon != 0:
                    valid_coords.append((lat, lon))
            except (ValueError, TypeError):
                continue
        
        if not valid_coords:
            return None
        
        lats = [coord[0] for coord in valid_coords]
        lons = [coord[1] for coord in valid_coords]
        
        return {
            "north": max(lats),
            "south": min(lats),
            "east": max(lons),
            "west": min(lons)
        }
    
    def process_query(self, query: str) -> Dict:
        """Main function to process user query and return map data"""
        # Parse the location query
        location_params = self.parse_location_query(query)
        
        # Query permits database
        permits = self.query_permits(location_params)
        
        # Create GeoJSON
        geojson = self.create_geojson(permits)
        
        # Calculate map bounds
        bounds = self.get_map_bounds(permits)
        
        # Create summary
        summary = self._create_summary(permits, location_params)
        
        return {
            "geojson": geojson,
            "bounds": bounds,
            "summary": summary,
            "location_params": location_params,
            "query": query
        }
    
    def _create_summary(self, permits: List[Dict], location_params: Dict) -> Dict:
        """Create a summary of the search results"""
        if not permits:
            return {
                "total_permits": 0,
                "message": "No permits found matching your criteria",
                "counties": {},
                "top_operators": {},
                "well_types": {},
                "date_range": None,
                "has_coordinates": 0
            }
        
        # Count by county
        counties = {}
        operators = {}
        well_types = {}
        
        for permit in permits:
            county = permit['county'] or 'Unknown'
            operator = permit['operator'] or 'Unknown'
            well_type = permit['well_type'] or 'Unknown'
            
            counties[county] = counties.get(county, 0) + 1
            operators[operator] = operators.get(operator, 0) + 1
            well_types[well_type] = well_types.get(well_type, 0) + 1
        
        # Get date range
        dates = [p['approval_date'] for p in permits if p['approval_date']]
        date_range = None
        if dates:
            dates.sort()
            date_range = {"earliest": dates[0], "latest": dates[-1]}
        
        return {
            "total_permits": len(permits),
            "counties": dict(sorted(counties.items(), key=lambda x: x[1], reverse=True)[:5]),
            "top_operators": dict(sorted(operators.items(), key=lambda x: x[1], reverse=True)[:5]),
            "well_types": well_types,
            "date_range": date_range,
            "has_coordinates": len([p for p in permits if p['latitude'] and p['longitude']])
        }

# Test the mapping agent
if __name__ == "__main__":
    mapper = DrillingPermitsMapper()
    
    # Test queries
    test_queries = [
        "Show me all drilling permits in Grady County",
        "Permits in section 15 township 15n range 24w",
        "Oil wells by MEWBOURNE OIL COMPANY",
        "Horizontal wells targeting Woodford formation"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Testing: '{query}'")
        result = mapper.process_query(query)
        
        print(f"   📊 Found {result['summary']['total_permits']} permits")
        print(f"   🗺️  {result['summary']['has_coordinates']} with coordinates")
        
        if result['summary']['total_permits'] > 0:
            print(f"   🏢 Top operators: {list(result['summary']['top_operators'].keys())[:3]}")
            print(f"   🗺️  Counties: {list(result['summary']['counties'].keys())[:3]}")
        
        print(f"   📁 GeoJSON features: {len(result['geojson']['features'])}")
