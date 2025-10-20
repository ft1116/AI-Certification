#!/usr/bin/env python3

import requests
import json

def test_complete_flow():
    """Test the complete mapping flow from backend to frontend"""
    
    print("🧪 Testing complete mapping flow...")
    
    # Test 1: Backend chat endpoint
    print("\n1️⃣ Testing backend chat endpoint...")
    try:
        response = requests.post("http://localhost:8003/chat", json={
            "query": "Map drilling permits in Grady County",
            "conversation_id": "debug123"
        })
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Backend response:")
            print(f"   - needs_mapping: {data.get('needs_mapping')}")
            print(f"   - mapping_data present: {data.get('mapping_data') is not None}")
            
            if data.get('mapping_data'):
                mapping_data = data['mapping_data']
                print(f"   - GeoJSON features: {len(mapping_data.get('features', []))}")
                if mapping_data.get('features'):
                    first_feature = mapping_data['features'][0]
                    coords = first_feature['geometry']['coordinates']
                    print(f"   - First feature coordinates: {coords}")
                    print(f"   - First feature properties: {first_feature['properties'].get('well_name', 'N/A')}")
        else:
            print(f"❌ Backend error: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"❌ Backend test error: {e}")
    
    # Test 2: Backend streaming endpoint
    print("\n2️⃣ Testing backend streaming endpoint...")
    try:
        response = requests.post("http://localhost:8003/chat/stream", 
                               json={"query": "Map drilling permits in Grady County", "conversation_id": "debug123"},
                               stream=True)
        
        if response.status_code == 200:
            print("✅ Streaming response received")
            mapping_data_found = False
            
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    if line_str.startswith('data: '):
                        try:
                            data_str = line_str[6:]  # Remove 'data: ' prefix
                            data = json.loads(data_str)
                            
                            if data.get('type') == 'done' and data.get('mapping_data'):
                                mapping_data_found = True
                                mapping_data = data['mapping_data']
                                print(f"   - Mapping data in stream: {len(mapping_data.get('features', []))} features")
                                break
                        except json.JSONDecodeError:
                            continue
            
            if not mapping_data_found:
                print("❌ No mapping data found in streaming response")
        else:
            print(f"❌ Streaming error: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Streaming test error: {e}")
    
    # Test 3: Frontend accessibility
    print("\n3️⃣ Testing frontend accessibility...")
    try:
        response = requests.get("http://localhost:3004")
        if response.status_code == 200:
            print("✅ Frontend is accessible")
        else:
            print(f"❌ Frontend error: {response.status_code}")
    except Exception as e:
        print(f"❌ Frontend test error: {e}")

if __name__ == "__main__":
    test_complete_flow()
