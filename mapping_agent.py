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
        
        # Oil & Gas activity weighting - higher numbers = more active
        # Note: Using database state names (Texas, OK, NM, etc.)
        self.state_activity_weights = {
            'Texas': 100,      # Most active oil & gas state
            'OK': 85,          # Oklahoma - very active, major shale plays
            'Oklahoma': 85,    # Alternative name
            'NM': 70,          # New Mexico - active, Permian Basin
            'New Mexico': 70,  # Alternative name
            'North Dakota': 60, # Bakken formation
            'Louisiana': 55,   # Haynesville, Gulf activity
            'Pennsylvania': 50, # Marcellus shale
            'Ohio': 45,        # Utica shale
            'West Virginia': 40, # Marcellus shale
            'Colorado': 35,    # DJ Basin, some activity
            'Wyoming': 30,     # Some activity
            'California': 25,  # Declining but still some activity
            'Alaska': 20,      # Remote but significant reserves
        }
        
    def parse_location_query(self, query: str) -> Dict:
        """Enhanced location parsing with fuzzy matching and better state recognition"""
        query_lower = query.lower()
        
        # Extract Section-Township-Range (STR) first
        section = self._extract_section(query_lower)
        township = self._extract_township(query_lower)
        range_val = self._extract_range(query_lower)
        
        # Enhanced state extraction with more patterns (run first to avoid conflicts)
        state = self._extract_state_enhanced(query_lower)
        
        # Enhanced county extraction with fuzzy matching
        county = self._extract_county_enhanced(query_lower)
        
        return {
            'county': county,
            'state': state,
            'section': section,
            'township': township,
            'range': range_val,
            'original_query': query
        }
    
    def _extract_section(self, query_lower: str) -> Optional[int]:
        """Extract section number"""
        section_patterns = [
            r'section\s+(\d+)',  # "Section 15"
            r'sec\s+(\d+)',      # "Sec 15"
            r's(\d+)',           # "S15"
        ]
        for pattern in section_patterns:
            section_match = re.search(pattern, query_lower)
            if section_match:
                return int(section_match.group(1))
        return None
    
    def _extract_township(self, query_lower: str) -> Optional[str]:
        """Extract township"""
        township_patterns = [
            r'township\s+(\d+[ns])',  # "Township 15N"
            r'twnshp\s+(\d+[ns])',    # "Twnshp 15N"
            r't(\d+[ns])',            # "T15N"
        ]
        for pattern in township_patterns:
            township_match = re.search(pattern, query_lower)
            if township_match:
                return township_match.group(1).upper()
        return None
    
    def _extract_range(self, query_lower: str) -> Optional[str]:
        """Extract range"""
        range_patterns = [
            r'range\s+(\d+[ew])',  # "Range 24W"
            r'rng\s+(\d+[ew])',    # "Rng 24W"
            r'r(\d+[ew])',         # "R24W"
        ]
        for pattern in range_patterns:
            range_match = re.search(pattern, query_lower)
            if range_match:
                return range_match.group(1).upper()
        return None
    
    def _extract_county_enhanced(self, query_lower: str) -> Optional[str]:
        """Enhanced county extraction with fuzzy matching"""
        # First try exact regex patterns
        county_patterns = [
            r'(\w+)\s+county',  # "Grady County"
            r'county\s+(\w+)',  # "county Grady"
            r'in\s+(\w+)\s+county',  # "in Grady County"
            r'(\w+)\s+co\s+(?!tx|texas|ok|oklahoma|nm|new mexico|colorado|nd|north dakota|la|louisiana|pa|pennsylvania|oh|ohio|wv|west virginia|wy|wyoming|ca|california|ak|alaska)',  # "Dallas Co" but not "Dallas Co TX"
        ]
        
        for pattern in county_patterns:
            county_match = re.search(pattern, query_lower)
            if county_match:
                potential_county = county_match.group(1).title()
                # Try to find exact match in database
                exact_match = self._find_exact_county_match(potential_county)
                if exact_match:
                    return exact_match
                
                # If no exact match, try to resolve ambiguity using oil & gas activity weighting
                resolved_county = self._resolve_ambiguous_county(potential_county)
                if resolved_county:
                    return resolved_county
        
        # If no exact match, try fuzzy matching for incomplete queries
        # Look for standalone county names (without "County" suffix)
        standalone_patterns = [
            r'in\s+(\w+)(?:\s|$)',  # "in Grady"
            r'activity\s+in\s+(\w+)',  # "activity in Grady"
            r'(\w+)(?:\s|$)',       # "Grady" at end
        ]
        
        for pattern in standalone_patterns:
            county_match = re.search(pattern, query_lower)
            if county_match:
                potential_county = county_match.group(1).title()
                # Skip common words and state names
                skip_words = ['the', 'a', 'an', 'and', 'or', 'in', 'on', 'at', 'to', 'for', 
                             'texas', 'oklahoma', 'new mexico', 'colorado', 'north dakota', 
                             'louisiana', 'pennsylvania', 'ohio', 'west virginia', 'wyoming', 
                             'california', 'alaska']
                if potential_county.lower() in skip_words:
                    continue
                
                # Try fuzzy match
                fuzzy_match = self._find_fuzzy_county_match(potential_county)
                if fuzzy_match:
                    return fuzzy_match
        
        return None
    
    def _extract_state_enhanced(self, query_lower: str) -> Optional[str]:
        """Enhanced state extraction with more patterns"""
        state_mappings = {
            'tx': 'Texas', 'texas': 'Texas',
            'ok': 'Oklahoma', 'oklahoma': 'Oklahoma', 
            'nm': 'New Mexico', 'new mexico': 'New Mexico',
            'co': 'Colorado', 'colorado': 'Colorado',
            'nd': 'North Dakota', 'north dakota': 'North Dakota',
            'la': 'Louisiana', 'louisiana': 'Louisiana',
            'pa': 'Pennsylvania', 'pennsylvania': 'Pennsylvania',
            'oh': 'Ohio', 'ohio': 'Ohio',
            'wv': 'West Virginia', 'west virginia': 'West Virginia',
            'wy': 'Wyoming', 'wyoming': 'Wyoming',
            'ca': 'California', 'california': 'California',
            'ak': 'Alaska', 'alaska': 'Alaska'
        }
        
        # Enhanced state patterns - be careful about "co" vs "county"
        state_patterns = [
            # County + State patterns (explicit "county" word)
            (r'(\w+)\s+county\s+(tx|texas|ok|oklahoma|nm|new mexico|colorado|nd|north dakota|la|louisiana|pa|pennsylvania|oh|ohio|wv|west virginia|wy|wyoming|ca|california|ak|alaska)', 2),
            # Simple county + state pattern (more flexible)
            (r'(\w+)\s+county\s+(texas|oklahoma|new mexico|colorado|north dakota|louisiana|pennsylvania|ohio|west virginia|wyoming|california|alaska)', 2),
            (r'county\s+(\w+)\s+(tx|texas|ok|oklahoma|nm|new mexico|colorado|nd|north dakota|la|louisiana|pa|pennsylvania|oh|ohio|wv|west virginia|wy|wyoming|ca|california|ak|alaska)', 2),
            
            # State + County patterns (explicit "county" word)
            (r'(tx|texas|ok|oklahoma|nm|new mexico|colorado|nd|north dakota|la|louisiana|pa|pennsylvania|oh|ohio|wv|west virginia|wy|wyoming|ca|california|ak|alaska)\s+(\w+)\s+county', 1),
            (r'(tx|texas|ok|oklahoma|nm|new mexico|colorado|nd|north dakota|la|louisiana|pa|pennsylvania|oh|ohio|wv|west virginia|wy|wyoming|ca|california|ak|alaska)\s+county\s+(\w+)', 1),
            
            # County abbreviation + State (like "Dallas Co TX")
            (r'(\w+)\s+co\s+(tx|texas|ok|oklahoma|nm|new mexico|colorado|nd|north dakota|la|louisiana|pa|pennsylvania|oh|ohio|wv|west virginia|wy|wyoming|ca|california|ak|alaska)', 2),
            (r'(tx|texas|ok|oklahoma|nm|new mexico|colorado|nd|north dakota|la|louisiana|pa|pennsylvania|oh|ohio|wv|west virginia|wy|wyoming|ca|california|ak|alaska)\s+(\w+)\s+co', 1),
            
            # County + State (no "County" word) - but avoid "co" here to prevent conflicts
            (r'(\w+)\s+(tx|texas|ok|oklahoma|nm|new mexico|colorado|nd|north dakota|la|louisiana|pa|pennsylvania|oh|ohio|wv|west virginia|wy|wyoming|ca|california|ak|alaska)', 2),
            (r'(tx|texas|ok|oklahoma|nm|new mexico|colorado|nd|north dakota|la|louisiana|pa|pennsylvania|oh|ohio|wv|west virginia|wy|wyoming|ca|california|ak|alaska)\s+(\w+)', 1),
            
            # Just state (be careful with "co" - only if it's clearly a state)
            (r'(tx|texas|ok|oklahoma|nm|new mexico|colorado|nd|north dakota|la|louisiana|pa|pennsylvania|oh|ohio|wv|west virginia|wy|wyoming|ca|california|ak|alaska)', 1),
            
            # State at end of query
            (r'(tx|texas|ok|oklahoma|nm|new mexico|colorado|nd|north dakota|la|louisiana|pa|pennsylvania|oh|ohio|wv|west virginia|wy|wyoming|ca|california|ak|alaska)$', 1),
            
            # Special case: "CO" as Colorado only when it's clearly a state (not county)
            (r'^(co|colorado)$', 1),  # Only at start/end of query
            (r'\s+(co|colorado)\s+', 1),  # Surrounded by spaces
            (r'\s+(co|colorado)$', 1),  # At end with space before
        ]
        
        for pattern, group_num in state_patterns:
            state_match = re.search(pattern, query_lower)
            if state_match:
                state_abbr = state_match.group(group_num).lower()
                return state_mappings.get(state_abbr)
        
        return None
    
    def _find_exact_county_match(self, county_name: str) -> Optional[str]:
        """Find exact county match in database"""
        if not Path(self.db_path).exists():
            return None
            
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT DISTINCT county FROM permits WHERE LOWER(county) = ?", (county_name.lower(),))
        result = cursor.fetchone()
        conn.close()
        
        return result[0] if result else None
    
    def _find_fuzzy_county_match(self, county_name: str, threshold: int = 80) -> Optional[str]:
        """Find fuzzy county match using string similarity with state activity weighting"""
        if not Path(self.db_path).exists():
            return None
            
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get counties with their states for weighting
        cursor.execute("""
            SELECT DISTINCT county, state 
            FROM permits 
            WHERE county IS NOT NULL AND county != '' AND state IS NOT NULL
        """)
        county_state_pairs = cursor.fetchall()
        conn.close()
        
        if not county_state_pairs:
            return None
        
        # Try exact match first
        exact_match = self._find_exact_county_match(county_name)
        if exact_match:
            return exact_match
        
        # Fuzzy matching with state activity weighting
        best_match = None
        best_weighted_score = 0
        
        for county, state in county_state_pairs:
            # Calculate string similarity score
            similarity_score = self._calculate_similarity(county_name.lower(), county.lower())
            
            if similarity_score >= threshold:
                # Get state activity weight (default to 10 for unknown states)
                state_weight = self.state_activity_weights.get(state, 10)
                
                # Calculate weighted score: similarity * (state_weight / 100)
                # This gives higher scores to counties in more active oil & gas states
                weighted_score = similarity_score * (state_weight / 100.0)
                
                if weighted_score > best_weighted_score:
                    best_weighted_score = weighted_score
                    best_match = county
        
        return best_match
    
    def _calculate_similarity(self, str1: str, str2: str) -> int:
        """Calculate simple string similarity percentage"""
        if not str1 or not str2:
            return 0
        
        # Simple character-based similarity
        matches = sum(1 for a, b in zip(str1, str2) if a == b)
        max_len = max(len(str1), len(str2))
        
        if max_len == 0:
            return 100
        
        return int((matches / max_len) * 100)
    
    def _resolve_ambiguous_county(self, county_name: str) -> Optional[str]:
        """Resolve ambiguous county names by preferring oil & gas active states"""
        if not Path(self.db_path).exists():
            return None
            
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Find all counties with this name across different states
        cursor.execute("""
            SELECT DISTINCT county, state, COUNT(*) as permit_count
            FROM permits 
            WHERE LOWER(county) = LOWER(?)
            GROUP BY county, state
            ORDER BY permit_count DESC
        """, (county_name,))
        
        results = cursor.fetchall()
        conn.close()
        
        if not results:
            return None
        
        if len(results) == 1:
            # Only one match, return it
            return results[0][0]
        
        # Multiple states have this county - prefer the one with highest oil & gas activity
        best_county = None
        best_score = 0
        
        for county, state, permit_count in results:
            # Combine state activity weight with permit count
            state_weight = self.state_activity_weights.get(state, 10)
            # Score = state_weight * log(permit_count + 1) to balance activity and data availability
            import math
            score = state_weight * math.log(permit_count + 1)
            
            if score > best_score:
                best_score = score
                best_county = county
        
        return best_county
        
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
