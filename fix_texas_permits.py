#!/usr/bin/env python3
"""
Fix Texas permits CSV parsing issues and convert county codes to names.
"""

import pandas as pd
import csv
import re
from pathlib import Path

def fix_texas_permits_csv():
    """Fix the Texas permits CSV parsing and county code issues."""
    
    input_file = "texas_permits_20251004.csv"
    output_file = "texas_permits_20251004_cleaned.csv"
    
    print(f"🔧 Fixing Texas permits CSV: {input_file}")
    print("=" * 60)
    
    # Texas county code to name mapping
    county_mapping = {
        '01': 'ANDERSON', '02': 'ANDREWS', '03': 'ANGELINA', '04': 'ARANSAS',
        '05': 'ARCHER', '06': 'ARMSTRONG', '07': 'ATASCOSA', '08': 'AUSTIN',
        '09': 'BAILEY', '10': 'BANDERA', '11': 'BASTROP', '12': 'BAYLOR',
        '13': 'BEE', '14': 'BELL', '15': 'BEXAR', '16': 'BLANCO',
        '17': 'BORDEN', '18': 'BOSQUE', '19': 'BOWIE', '20': 'BRAZORIA',
        '21': 'BRAZOS', '22': 'BREWSTER', '23': 'BRISCOE', '24': 'BROOKS',
        '25': 'BROWN', '26': 'BURLESON', '27': 'BURNET', '28': 'CALDWELL',
        '29': 'CALHOUN', '30': 'CALLAHAN', '31': 'CAMERON', '32': 'CAMP',
        '33': 'CARSON', '34': 'CASS', '35': 'CASTRO', '36': 'CHAMBERS',
        '37': 'CHEROKEE', '38': 'CHILDRESS', '39': 'CLAY', '40': 'COCHRAN',
        '41': 'COKE', '42': 'COLEMAN', '43': 'COLLIN', '44': 'COLLINGSWORTH',
        '45': 'COLORADO', '46': 'COMAL', '47': 'COMANCHE', '48': 'CONCHO',
        '49': 'COOKE', '50': 'CORYELL', '51': 'COTTLE', '52': 'CRANE',
        '53': 'CROCKETT', '54': 'CROSBY', '55': 'CULBERSON', '56': 'DALLAM',
        '57': 'DALLAS', '58': 'DAWSON', '59': 'DEAF SMITH', '60': 'DELTA',
        '61': 'DENTON', '62': 'DEWITT', '63': 'DICKENS', '64': 'DIMMIT',
        '65': 'DONLEY', '66': 'DUVAL', '67': 'EASTLAND', '68': 'ECTOR',
        '69': 'EDWARDS', '70': 'ELLIS', '71': 'EL PASO', '72': 'ERATH',
        '73': 'FALLS', '74': 'FANNIN', '75': 'FAYETTE', '76': 'FISHER',
        '77': 'FLOYD', '78': 'FOARD', '79': 'FORT BEND', '80': 'FRANKLIN',
        '81': 'FREESTONE', '82': 'FRIO', '83': 'GAINES', '84': 'GALVESTON',
        '85': 'GARZA', '86': 'GILLESPIE', '87': 'GLASSCOCK', '88': 'GOLIAD',
        '89': 'GONZALES', '90': 'GRAY', '91': 'GRAYSON', '92': 'GREGG',
        '93': 'GRIMES', '94': 'GUADALUPE', '95': 'HALE', '96': 'HALL',
        '97': 'HAMILTON', '98': 'HANSFORD', '99': 'HARDEMAN', '100': 'HARDIN',
        '101': 'HARRIS', '102': 'HARRISON', '103': 'HARTLEY', '104': 'HASKELL',
        '105': 'HAYS', '106': 'HEMPHILL', '107': 'HENDERSON', '108': 'HIDALGO',
        '109': 'HILL', '110': 'HOCKLEY', '111': 'HOOD', '112': 'HOPKINS',
        '113': 'HOUSTON', '114': 'HOWARD', '115': 'HUDSPETH', '116': 'HUNT',
        '117': 'HUTCHINSON', '118': 'IRION', '119': 'JACK', '120': 'JACKSON',
        '121': 'JASPER', '122': 'JEFF DAVIS', '123': 'JEFFERSON', '124': 'JIM HOGG',
        '125': 'JIM WELLS', '126': 'JOHNSON', '127': 'JONES', '128': 'KARNES',
        '129': 'KAUFMAN', '130': 'KENDALL', '131': 'KENEDY', '132': 'KENT',
        '133': 'KERR', '134': 'KIMBLE', '135': 'KING', '136': 'KINNEY',
        '137': 'KLEBERG', '138': 'KNOX', '139': 'LAMAR', '140': 'LAMB',
        '141': 'LAMPASAS', '142': 'LA SALLE', '143': 'LAVACA', '144': 'LEE',
        '145': 'LEON', '146': 'LIBERTY', '147': 'LIMESTONE', '148': 'LIPSCOMB',
        '149': 'LIVE OAK', '150': 'LLANO', '151': 'LOVING', '152': 'LUBBOCK',
        '153': 'LYNN', '154': 'MCCULLOCH', '155': 'MCLENNAN', '156': 'MCMULLEN',
        '157': 'MADISON', '158': 'MARION', '159': 'MARTIN', '160': 'MASON',
        '161': 'MATAGORDA', '162': 'MAVERICK', '163': 'MEDINA', '164': 'MENARD',
        '165': 'MIDLAND', '166': 'MILAM', '167': 'MILLS', '168': 'MITCHELL',
        '169': 'MONTAGUE', '170': 'MONTGOMERY', '171': 'MOORE', '172': 'MORRIS',
        '173': 'MOTLEY', '174': 'NACOGDOCHES', '175': 'NAVARRO', '176': 'NEWTON',
        '177': 'NOLAN', '178': 'NUECES', '179': 'OCHILTREE', '180': 'OLDHAM',
        '181': 'ORANGE', '182': 'PALO PINTO', '183': 'PANOLA', '184': 'PARKER',
        '185': 'PARMER', '186': 'PECOS', '187': 'POLK', '188': 'POTTER',
        '189': 'PRESIDIO', '190': 'RAINS', '191': 'RANDALL', '192': 'REAGAN',
        '193': 'REAL', '194': 'RED RIVER', '195': 'REEVES', '196': 'REFUGIO',
        '197': 'ROBERTS', '198': 'ROBERTSON', '199': 'ROCKWALL', '200': 'RUNNELS',
        '201': 'RUSK', '202': 'SABINE', '203': 'SAN AUGUSTINE', '204': 'SAN JACINTO',
        '205': 'SAN PATRICIO', '206': 'SAN SABA', '207': 'SCHLEICHER', '208': 'SCURRY',
        '209': 'SHACKELFORD', '210': 'SHELBY', '211': 'SHERMAN', '212': 'SMITH',
        '213': 'SOMERVELL', '214': 'STARR', '215': 'STEPHENS', '216': 'STERLING',
        '217': 'STONEWALL', '218': 'SUTTON', '219': 'SWISHER', '220': 'TARRANT',
        '221': 'TAYLOR', '222': 'TERRELL', '223': 'TERRY', '224': 'THROCKMORTON',
        '225': 'TITUS', '226': 'TOM GREEN', '227': 'TRAVIS', '228': 'TRINITY',
        '229': 'TYLER', '230': 'UPSHUR', '231': 'UPTON', '232': 'UVALDE',
        '233': 'VAL VERDE', '234': 'VAN ZANDT', '235': 'VICTORIA', '236': 'WALKER',
        '237': 'WALLER', '238': 'WARD', '239': 'WASHINGTON', '240': 'WEBB',
        '241': 'WHARTON', '242': 'WHEELER', '243': 'WICHITA', '244': 'WILBARGER',
        '245': 'WILLACY', '246': 'WILLIAMSON', '247': 'WILSON', '248': 'WINKLER',
        '249': 'WISE', '250': 'WOOD', '251': 'YOAKUM', '252': 'YOUNG',
        '253': 'ZAPATA', '254': 'ZAVALA',
        # Handle the letter codes we saw
        '7C': 'IRION',  # From the diff output
    }
    
    try:
        # Read the CSV with proper handling of quoted fields
        print("📖 Reading CSV with proper quote handling...")
        
        # First, let's try to read it with pandas and see what we get
        df = pd.read_csv(input_file, quotechar='"', escapechar='\\')
        
        print(f"✅ Loaded {len(df)} records")
        print(f"📊 Columns: {list(df.columns)}")
        
        # Check the county column
        print(f"\n🔍 County column sample:")
        print(df['County'].head(10).tolist())
        
        # Check if we have the county names in the data
        print(f"\n🔍 Looking for county names in the data...")
        
        # Let's examine a few rows to understand the structure
        print(f"\n📋 Sample row data:")
        for i in range(min(3, len(df))):
            print(f"Row {i}:")
            print(f"  County: {df.iloc[i]['County']}")
            print(f"  Operator: {df.iloc[i]['Operator']}")
            print(f"  Well_Number: {df.iloc[i]['Well_Number']}")
            print()
        
        # Try to find county names in the data
        # Looking at the diff output, it seems like the county names might be in the Operator field
        # or there might be a parsing issue
        
        # Let's check if there are any county names in the data
        all_text = ' '.join(df.astype(str).values.flatten())
        found_counties = []
        for code, name in county_mapping.items():
            if name in all_text:
                found_counties.append((code, name))
        
        print(f"🏛️ Found county names in data: {found_counties}")
        
        # Create a mapping function
        def map_county_code(code):
            """Map county code to county name."""
            if pd.isna(code):
                return 'UNKNOWN'
            
            code_str = str(code).strip()
            
            # Handle the specific codes we saw
            if code_str in county_mapping:
                return county_mapping[code_str]
            
            # If it's already a name, return as is
            if code_str.upper() in county_mapping.values():
                return code_str.upper()
            
            return code_str  # Return original if no mapping found
        
        # Apply the mapping
        print(f"\n🔄 Converting county codes to names...")
        df['County_Name'] = df['County'].apply(map_county_code)
        
        # Add state field
        df['State'] = 'Texas'
        
        # Show results
        print(f"\n📊 County conversion results:")
        county_counts = df['County_Name'].value_counts().head(10)
        print(county_counts)
        
        # Save the cleaned data
        print(f"\n💾 Saving cleaned data to {output_file}...")
        df.to_csv(output_file, index=False)
        
        print(f"✅ Successfully created cleaned file: {output_file}")
        print(f"📊 Total records: {len(df)}")
        print(f"🏛️ Unique counties: {df['County_Name'].nunique()}")
        
        return output_file
        
    except Exception as e:
        print(f"❌ Error processing file: {e}")
        return None

def main():
    """Main function."""
    print("🚀 Texas Permits CSV Fixer")
    print("=" * 60)
    
    result = fix_texas_permits_csv()
    
    if result:
        print(f"\n🎉 Success! Fixed file created: {result}")
        print("\n📋 Next steps:")
        print("1. Update your ingestion script to use the cleaned file")
        print("2. The cleaned file has County_Name and State fields")
        print("3. Run the full ingestion with the cleaned data")
    else:
        print("\n❌ Failed to fix the file. Check the error messages above.")

if __name__ == "__main__":
    main()

