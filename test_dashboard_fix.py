#!/usr/bin/env python3
"""
Simple test for the dashboard fix - run from project root
"""

def test_api_request():
    """Test the API request directly to isolate the issue."""
    
    print("🧪 Testing API request with your exact parameters...")
    
    import requests
    from config import coordinate_converter, query_builder
    
    # Your exact parameters
    latitude = 23.564
    longitude = -72.773
    start_date = "1955-01-01"
    end_date = "1960-12-31"
    
    print(f"Parameters:")
    print(f"  Location: {latitude}°N, {longitude}°W")
    print(f"  Date range: {start_date} to {end_date}")
    
    try:
        # Step 1: Convert coordinates
        lat_idx = coordinate_converter.lat_to_grid_index(latitude)
        lon_idx = coordinate_converter.lon_to_grid_index(longitude)
        start_time_idx = coordinate_converter.date_to_time_index(start_date)
        end_time_idx = coordinate_converter.date_to_time_index(end_date)
        
        print(f"\nGrid indices:")
        print(f"  Lat: {latitude}°N → {lat_idx}")
        print(f"  Lon: {longitude}°W → {lon_idx}")
        print(f"  Time: {start_date} to {end_date} → {start_time_idx}:{end_time_idx}")
        
        # Step 2: Build URL
        url = query_builder.build_query_url(latitude, longitude, start_date, end_date)
        print(f"\nGenerated URL:")
        print(f"  {url}")
        
        # Step 3: Make request
        print(f"\n🌐 Making HTTP request...")
        response = requests.get(url, timeout=30)
        
        print(f"Response status: {response.status_code}")
        
        if response.status_code == 200:
            print(f"✅ HTTP request successful!")
            print(f"Content length: {len(response.text)} characters")
            print(f"Content type: {response.headers.get('content-type', 'unknown')}")
            
            # Step 4: Test pandas parsing
            print(f"\n📊 Testing pandas parsing...")
            
            import pandas as pd
            import io
            
            # Show first few lines of response
            lines = response.text.split('\n')[:5]
            print(f"First few lines of response:")
            for i, line in enumerate(lines):
                print(f"  {i}: {line}")
            
            # Try parsing
            df = pd.read_csv(io.StringIO(response.text), skipinitialspace=True)
            
            print(f"✅ Pandas parsing successful!")
            print(f"DataFrame shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            
            if not df.empty:
                print(f"Sample data:")
                print(df.head())
                return True
            else:
                print(f"❌ DataFrame is empty")
                return False
        else:
            print(f"❌ HTTP request failed with status {response.status_code}")
            print(f"Response text: {response.text[:500]}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_extractor():
    """Test the extractor class directly."""
    
    print(f"\n🧪 Testing ERDDAPExtractor class...")
    
    try:
        from pipeline.extract import ERDDAPExtractor
        
        extractor = ERDDAPExtractor()
        
        # Your exact parameters
        latitude = 23.564
        longitude = -72.773
        start_date = "1955-01-01"
        end_date = "1960-12-31"
        
        print(f"🌊 Calling fetch_data_for_location...")
        
        df, metadata = extractor.fetch_data_for_location(
            latitude=latitude,
            longitude=longitude,
            start_date=start_date,
            end_date=end_date
        )
        
        if df is not None:
            print(f"✅ Extractor successful!")
            print(f"Retrieved {len(df)} rows")
            print(f"Data source: {metadata.get('data_source', 'unknown')}")
            return True
        else:
            print(f"❌ Extractor returned None")
            return False
            
    except Exception as e:
        print(f"❌ Extractor error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run the tests."""
    print("🔧 Testing Dashboard Fix")
    print("=" * 40)
    
    # Test 1: Direct API request
    api_success = test_api_request()
    
    if api_success:
        # Test 2: Extractor class
        extractor_success = test_extractor()
        
        if extractor_success:
            print(f"\n🎉 All tests passed!")
            print(f"✅ The dashboard should now work!")
            print(f"🚀 Try: streamlit run dashboard/app.py")
        else:
            print(f"\n⚠️ API works but extractor has issues")
    else:
        print(f"\n❌ Basic API request failed")
        print(f"🔍 Check coordinate conversion and URL generation")

if __name__ == "__main__":
    main()