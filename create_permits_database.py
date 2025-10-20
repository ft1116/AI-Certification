#!/usr/bin/env python3
"""
Create SQLite database from CSV permit data for the mapping agent
"""

import sqlite3
import pandas as pd
import os
from pathlib import Path

def create_permits_database():
    """Create SQLite database from CSV files"""
    
    # Create the directory structure
    db_dir = Path("Drilling Permits/data")
    db_dir.mkdir(parents=True, exist_ok=True)
    db_path = db_dir / "permits.db"
    
    # Remove existing database if it exists
    if db_path.exists():
        db_path.unlink()
    
    # Create database connection
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create permits table
    cursor.execute('''
        CREATE TABLE permits (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            api_number TEXT,
            entity_name TEXT,
            well_name TEXT,
            well_number TEXT,
            state TEXT,
            county TEXT,
            section REAL,
            township TEXT,
            range_val TEXT,
            pbh_section REAL,
            pbh_township TEXT,
            pbh_range TEXT,
            formation_name TEXT,
            formation_depth REAL,
            total_depth REAL,
            measured_total_depth REAL,
            true_vertical_depth REAL,
            well_type TEXT,
            well_status TEXT,
            permit_type TEXT,
            well_class TEXT,
            surf_long_x REAL,
            surf_lat_y REAL,
            proposed_bottom_hole_long_x REAL,
            proposed_bottom_hole_lat_y REAL
        )
    ''')
    
    # Load and insert Oklahoma permits
    if os.path.exists("oklahoma_permits_streamlined.csv"):
        print("Loading Oklahoma permits...")
        ok_df = pd.read_csv("oklahoma_permits_streamlined.csv")
        
        # Rename 'Range' column to 'range_val' to avoid SQL keyword conflict
        if 'Range' in ok_df.columns:
            ok_df = ok_df.rename(columns={'Range': 'range_val'})
        if 'PBH_Range' in ok_df.columns:
            ok_df = ok_df.rename(columns={'PBH_Range': 'pbh_range'})
        
        ok_df.to_sql('permits', conn, if_exists='append', index=False)
        print(f"Loaded {len(ok_df)} Oklahoma permits")
    
    # Load and insert Texas permits with column mapping
    if os.path.exists("texas_permits_20251004_cleaned.csv"):
        print("Loading Texas permits...")
        tx_df = pd.read_csv("texas_permits_20251004_cleaned.csv")
        
        # Map Texas CSV columns to match the database schema
        tx_mapped = pd.DataFrame()
        tx_mapped['api_number'] = tx_df['API_Number']
        tx_mapped['entity_name'] = tx_df['Operator']
        tx_mapped['well_name'] = tx_df['Lease_Name']
        tx_mapped['well_number'] = tx_df['Well_Number']
        tx_mapped['state'] = tx_df['State']
        tx_mapped['county'] = tx_df['County']
        tx_mapped['section'] = tx_df['Section']
        tx_mapped['township'] = None  # Not available in Texas data
        tx_mapped['range_val'] = None  # Not available in Texas data
        tx_mapped['pbh_section'] = None  # Not available in Texas data
        tx_mapped['pbh_township'] = None  # Not available in Texas data
        tx_mapped['pbh_range'] = None  # Not available in Texas data
        tx_mapped['formation_name'] = None  # Not available in Texas data
        tx_mapped['formation_depth'] = None  # Not available in Texas data
        tx_mapped['total_depth'] = None  # Not available in Texas data
        tx_mapped['measured_total_depth'] = None  # Not available in Texas data
        tx_mapped['true_vertical_depth'] = None  # Not available in Texas data
        tx_mapped['well_type'] = tx_df['Wellbore_Profile']
        tx_mapped['well_status'] = tx_df['Filing_Purpose']
        tx_mapped['permit_type'] = tx_df['Filing_Purpose']
        tx_mapped['well_class'] = None  # Not available in Texas data
        tx_mapped['surf_long_x'] = tx_df['Longitude']
        tx_mapped['surf_lat_y'] = tx_df['Latitude']
        tx_mapped['proposed_bottom_hole_long_x'] = None  # Not available in Texas data
        tx_mapped['proposed_bottom_hole_lat_y'] = None  # Not available in Texas data
        
        tx_mapped.to_sql('permits', conn, if_exists='append', index=False)
        print(f"Loaded {len(tx_mapped)} Texas permits")
    
    # Create indexes for better performance
    cursor.execute('CREATE INDEX idx_county ON permits(county)')
    cursor.execute('CREATE INDEX idx_state ON permits(state)')
    cursor.execute('CREATE INDEX idx_entity_name ON permits(entity_name)')
    cursor.execute('CREATE INDEX idx_well_type ON permits(well_type)')
    cursor.execute('CREATE INDEX idx_coordinates ON permits(surf_lat_y, surf_long_x)')
    
    conn.commit()
    conn.close()
    
    print(f"✅ Database created successfully at: {db_path}")
    
    # Show some stats
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Count total permits
    cursor.execute('SELECT COUNT(*) FROM permits')
    total = cursor.fetchone()[0]
    print(f"Total permits: {total}")
    
    # Count by state
    cursor.execute('SELECT state, COUNT(*) FROM permits GROUP BY state')
    states = cursor.fetchall()
    for state, count in states:
        print(f"{state}: {count} permits")
    
    # Count by county (top 10)
    cursor.execute('SELECT county, state, COUNT(*) FROM permits GROUP BY county, state ORDER BY COUNT(*) DESC LIMIT 10')
    counties = cursor.fetchall()
    print("\nTop 10 counties by permit count:")
    for county, state, count in counties:
        print(f"{county}, {state}: {count} permits")
    
    conn.close()

if __name__ == "__main__":
    create_permits_database()
