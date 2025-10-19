#!/usr/bin/env python3
"""
Streamline Oklahoma permits data by keeping only essential columns for mineral insights.
Reduces from 134 columns to 20 columns (85% reduction).
"""

import pandas as pd
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OklahomaPermitsStreamliner:
    """Streamline Oklahoma permits data for faster ingestion."""
    
    def __init__(self, input_file: str, output_file: str):
        self.input_file = input_file
        self.output_file = output_file
        
        # Essential columns to keep (20 out of 134)
        self.essential_columns = [
            # Core Identification
            'API_Number',           # 1 - Unique identifier
            'Entity_Name',          # 3 - Operator name
            'Well_Name',            # 9 - Well name
            'Well_Number',          # 10 - Well number
            
            # Location Data
            'State',                # 7 - Always Oklahoma
            'County',               # 17 - Primary location
            'Section',              # 18 - Surface section
            'Township',             # 19 - Surface township
            'Range',                # 20 - Surface range
            'PBH_Section',          # 33 - Bottom hole section
            'PBH_Township',         # 34 - Bottom hole township
            'PBH_Range',            # 35 - Bottom hole range
            
            # Geological Data
            'Formation_Name',       # 50 - Target formation
            'Formation_Depth',      # 51 - Target depth
            'Total_Depth',          # 52 - Well depth
            'Measured_Total_Depth', # 46 - Measured depth
            'True_Vertical_Depth',  # 47 - Vertical depth
            
            # Well Details
            'Well_Type',            # 11 - Type of well
            'Well_Status',          # 12 - Current status
            'Permit_Type',          # 57 - Type of permit
            'Well_Class',           # 89 - Well classification
            
            # Coordinates (useful for mapping)
            'Surf_Long_X',          # 15 - Surface longitude
            'Surf_Lat_Y',           # 16 - Surface latitude
            'Proposed_Bottom_Hole_Long_X',  # 30 - Bottom hole longitude
            'Proposed_Bottom_Hole_Lat_Y',   # 31 - Bottom hole latitude
        ]

    def streamline(self):
        """Streamline the Oklahoma permits data."""
        logger.info(f"Streamlining Oklahoma permits: {self.input_file}")
        
        if not Path(self.input_file).exists():
            logger.error(f"Input file not found: {self.input_file}")
            return False
        
        try:
            # Read the full CSV
            logger.info("Reading full Oklahoma permits data...")
            df = pd.read_csv(self.input_file, dtype=str)
            logger.info(f"Loaded {len(df)} records with {len(df.columns)} columns")
            
            # Check which essential columns exist
            missing_columns = [col for col in self.essential_columns if col not in df.columns]
            if missing_columns:
                logger.warning(f"Missing columns: {missing_columns}")
            
            # Keep only essential columns that exist
            available_columns = [col for col in self.essential_columns if col in df.columns]
            logger.info(f"Keeping {len(available_columns)} essential columns")
            
            # Create streamlined dataframe
            df_streamlined = df[available_columns].copy()
            
            # Add some basic data quality info
            logger.info("\n📊 Data Quality Summary:")
            logger.info(f"Total records: {len(df_streamlined)}")
            logger.info(f"Columns kept: {len(available_columns)}")
            logger.info(f"Reduction: {len(df.columns)} → {len(available_columns)} columns ({100 * (1 - len(available_columns)/len(df.columns)):.1f}% reduction)")
            
            # Show sample of key fields
            logger.info("\n🔍 Sample Data:")
            sample_cols = ['API_Number', 'Entity_Name', 'County', 'Formation_Name', 'Formation_Depth']
            available_sample_cols = [col for col in sample_cols if col in df_streamlined.columns]
            if available_sample_cols:
                logger.info(df_streamlined[available_sample_cols].head(3).to_string(index=False))
            
            # Save streamlined data
            df_streamlined.to_csv(self.output_file, index=False)
            logger.info(f"✅ Successfully created streamlined file: {self.output_file}")
            
            # File size comparison
            original_size = Path(self.input_file).stat().st_size
            new_size = Path(self.output_file).stat().st_size
            size_reduction = 100 * (1 - new_size / original_size)
            logger.info(f"📁 File size: {original_size:,} → {new_size:,} bytes ({size_reduction:.1f}% reduction)")
            
            return True
            
        except Exception as e:
            logger.error(f"Error streamlining data: {e}")
            return False

def main():
    """Main function."""
    input_file = 'oklahoma_permits.csv'
    output_file = 'oklahoma_permits_streamlined.csv'
    
    streamliner = OklahomaPermitsStreamliner(input_file, output_file)
    success = streamliner.streamline()
    
    if success:
        print("\n🎉 Oklahoma permits data successfully streamlined!")
        print(f"📁 Input:  {input_file} (134 columns)")
        print(f"📁 Output: {output_file} (20 columns)")
        print("🚀 Ready for fast ingestion!")
    else:
        print("\n❌ Streamlining failed!")

if __name__ == "__main__":
    main()

