#!/usr/bin/env python3
"""
Enhanced Location Parser
Implements improvements based on comprehensive testing analysis.
"""

import re
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from rapidfuzz import fuzz, process

class EnhancedLocationParser:
    def __init__(self, db_path: str = "Drilling Permits/data/permits.db"):
        self.db_path = db_path
        self.county_list = self._load_county_database()
        self.state_mappings = {
            'tx': 'Texas', 'texas': 'Texas',
            'ok': 'Oklahoma', 'oklahoma': 'Oklahoma', 
            'nm': 'New Mexico', 'new mexico': 'New Mexico'
        }
        
    def _load_county_database(self) -> List[str]:
        """Load all counties from the database for fuzzy matching"""
        if not Path(self.db_path).exists():
            return []
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT DISTINCT county 
            FROM permits 
            WHERE county IS NOT NULL AND county != ''
            ORDER BY county
        """)
        
        counties = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        return counties
    
    def parse_location_query(self, query: str) -> Dict:
        """Enhanced location parsing with fuzzy matching and better state recognition"""
        query_lower = query.lower()
        
        # Extract Section-Township-Range (STR) first (already working perfectly)
        section = self._extract_section(query_lower)
        township = self._extract_township(query_lower)
        range_val = self._extract_range(query_lower)
        
        # Enhanced county extraction with fuzzy matching
        county = self._extract_county_enhanced(query_lower)
        
        # Enhanced state extraction with more patterns
        state = self._extract_state_enhanced(query_lower)
        
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
        ]
        
        for pattern in county_patterns:
            county_match = re.search(pattern, query_lower)
            if county_match:
                potential_county = county_match.group(1).title()
                # Try to find exact match in database
                exact_match = self._find_exact_county_match(potential_county)
                if exact_match:
                    return exact_match
        
        # If no exact match, try fuzzy matching for incomplete queries
        # Look for standalone county names (without "County" suffix)
        standalone_patterns = [
            r'in\s+(\w+)(?:\s|$)',  # "in Grady"
            r'(\w+)(?:\s|$)',       # "Grady" at end
        ]
        
        for pattern in standalone_patterns:
            county_match = re.search(pattern, query_lower)
            if county_match:
                potential_county = county_match.group(1).title()
                # Skip common words
                if potential_county.lower() in ['the', 'a', 'an', 'and', 'or', 'in', 'on', 'at', 'to', 'for']:
                    continue
                
                # Try fuzzy match
                fuzzy_match = self._find_fuzzy_county_match(potential_county)
                if fuzzy_match:
                    return fuzzy_match
        
        return None
    
    def _extract_state_enhanced(self, query_lower: str) -> Optional[str]:
        """Enhanced state extraction with more patterns"""
        # Enhanced state patterns
        state_patterns = [
            # County + State patterns
            (r'(\w+)\s+county\s+(tx|texas|ok|oklahoma|nm|new mexico)', 2),
            (r'county\s+(\w+)\s+(tx|texas|ok|oklahoma|nm|new mexico)', 2),
            
            # State + County patterns
            (r'(tx|texas|ok|oklahoma|nm|new mexico)\s+(\w+)\s+county', 1),
            (r'(tx|texas|ok|oklahoma|nm|new mexico)\s+county\s+(\w+)', 1),
            
            # County + State (no "County" word)
            (r'(\w+)\s+(tx|texas|ok|oklahoma|nm|new mexico)', 2),
            (r'(tx|texas|ok|oklahoma|nm|new mexico)\s+(\w+)', 1),
            
            # Just state
            (r'(tx|texas|ok|oklahoma|nm|new mexico)', 1),
            
            # State at end of query
            (r'(tx|texas|ok|oklahoma|nm|new mexico)$', 1),
        ]
        
        for pattern, group_num in state_patterns:
            state_match = re.search(pattern, query_lower)
            if state_match:
                state_abbr = state_match.group(group_num).lower()
                return self.state_mappings.get(state_abbr)
        
        return None
    
    def _find_exact_county_match(self, county_name: str) -> Optional[str]:
        """Find exact county match in database"""
        for county in self.county_list:
            if county.lower() == county_name.lower():
                return county
        return None
    
    def _find_fuzzy_county_match(self, county_name: str, threshold: int = 80) -> Optional[str]:
        """Find fuzzy county match using rapidfuzz"""
        if not self.county_list:
            return None
        
        # Try exact match first
        exact_match = self._find_exact_county_match(county_name)
        if exact_match:
            return exact_match
        
        # Try fuzzy match
        fuzzy_match = process.extractOne(
            county_name, 
            self.county_list, 
            scorer=fuzz.ratio
        )
        
        if fuzzy_match and fuzzy_match[1] >= threshold:
            return fuzzy_match[0]
        
        return None
    
    def test_database_query(self, parsed_location: Dict) -> Dict:
        """Test what the database query would return with enhanced matching"""
        if not Path(self.db_path).exists():
            return {'found_permits': 0, 'error': 'Database not found'}
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Build enhanced query with fuzzy matching
        conditions = []
        params = []
        
        if parsed_location['county']:
            # Use case-insensitive matching
            conditions.append("LOWER(county) LIKE ?")
            params.append(f"%{parsed_location['county'].lower()}%")
        
        if parsed_location['state']:
            # Use case-insensitive matching
            conditions.append("LOWER(state) LIKE ?")
            params.append(f"%{parsed_location['state'].lower()}%")
        
        if parsed_location['section']:
            conditions.append("section = ?")
            params.append(parsed_location['section'])
        
        if parsed_location['township']:
            conditions.append("LOWER(township) LIKE ?")
            params.append(f"%{parsed_location['township'].lower()}%")
        
        if parsed_location['range']:
            conditions.append("LOWER(range_val) LIKE ?")
            params.append(f"%{parsed_location['range'].lower()}%")
        
        if not conditions:
            conditions.append("1=1")
        
        sql = f"""
            SELECT COUNT(*) as count, 
                   GROUP_CONCAT(DISTINCT county) as counties,
                   GROUP_CONCAT(DISTINCT state) as states
            FROM permits 
            WHERE {' AND '.join(conditions)}
        """
        
        cursor.execute(sql, params)
        result = cursor.fetchone()
        
        conn.close()
        
        return {
            'found_permits': result[0] if result else 0,
            'matching_counties': result[1].split(',') if result[1] else [],
            'matching_states': result[2].split(',') if result[2] else []
        }

def test_enhanced_parser():
    """Test the enhanced parser with some example queries"""
    parser = EnhancedLocationParser()
    
    test_queries = [
        "Map drilling permits in Grady County",
        "Show me permits in Grady County OK", 
        "Any drilling activity in Grady?",
        "Map permits in Leon County TX",
        "Show drilling in Leon",
        "Activity in Dawson County",
        "Permits in Texas",
        "Drilling in Oklahoma",
        "Map permits in Section 15 Township 15N Range 24W",
        "Show drilling in Grady County Section 10 Township 8S Range 12E",
    ]
    
    print("🧪 Testing Enhanced Location Parser")
    print("=" * 50)
    
    for query in test_queries:
        parsed = parser.parse_location_query(query)
        db_result = parser.test_database_query(parsed)
        
        print(f"\nQuery: {query}")
        print(f"Parsed: {parsed}")
        print(f"Database: {db_result['found_permits']} permits found")
        if db_result['matching_counties']:
            print(f"Counties: {', '.join(db_result['matching_counties'])}")
        if db_result['matching_states']:
            print(f"States: {', '.join(db_result['matching_states'])}")

if __name__ == "__main__":
    test_enhanced_parser()
