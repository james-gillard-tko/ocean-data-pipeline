#!/usr/bin/env python3
"""
Test with the exact parameters that caused the error
"""

from pipeline.extract import ERDDAPExtractor
import traceback

def test_exact_parameters():
    """Test with the exact parameters from the dashboard."""
    
    print("🧪 Testing exact parameters that caused the error...")
    
    # Exact parameters from your test
    latitude = 23.564
    longitude = -72.773
    start_date = "1955-01-01"
    end_date = "1960-12-31"
    
    print(f"Parameters:")
    print(f"  Location: {latitude}°N, {longitude}°W")
    print(f"  Date range: {start_date} to {end_date}")
    
    try:
        # Create extractor
        extractor = ERDDAPExtractor()
        
        print(f"\n🌊 Fetching data...")
        
        # This should reproduce the exact error
        df, metadata = extractor.fetch_data_for_location(
            latitude=latitude,
            longitude=longitude,
            start_date=start_date,
            end_date=end_date
        )
        
        if df is not None and not df.empty:
            print(f"✅ Success! Retrieved {len(df)} rows")
            print(f"📊 Data source: {metadata.get('data_source', 'unknown')}")
            print(f"🎯 Quality score: {metadata.get('quality_score', 'N/A')}")
            
            print(f"\nSample data:")
            print(df.head())
            
            print(f"\nData columns: {list(df.columns)}")
            print(f"Data types:")
            print(df.dtypes)
            
            return True
            
        else:
            print(f"❌ No data returned")
            return False
            
    except Exception as e:
        print(f"❌ Error occurred: {str(e)}")
        print(f"\nFull traceback:")
        traceback.print_exc()
        return False

def test_step_by_step():
    """Test the process step by step to identify where it fails."""
    
    print(f"\n🔍 Testing step by step...")
    
    try:
        from config import coordinate_converter, query_builder
        
        # Step 1: Coordinate conversion
        latitude = 23.564
        longitude = -72.773
        
        lat_idx = coordinate_converter.lat_to_grid_index(latitude)
        lon_idx = coordinate_converter.lon_to_grid_index(longitude)
        
        print(f"Step 1 - Coordinate conversion:")
        print(f"  {latitude}°N, {longitude}°W → grid[{lat_idx}, {lon_idx}]")
        
        # Step 2: Date conversion
        start_date = "1955-01-01"
        end_date = "1960-12-31"
        
        start_time_idx = coordinate_converter.date_to_time_index(start_date)
        end_time_idx = coordinate_converter.date_to_time_index(end_date)
        
        print(f"Step 2 - Date conversion:")
        print(f"  {start_date} to {end_date} → time[{start_time_idx}:{end_time_idx}]")
        
        # Step 3: URL generation
        url = query_builder.build_query_url(latitude, longitude, start_date, end_date)
        print(f"Step 3 - URL generation:")
        print(f"  {url}")
        
        # Step 4: HTTP request
        import requests
        print(f"Step 4 - HTTP request...")
        
        response = requests.get(url, timeout=15)
        print(f"  Status: {response.status_code}")
        print(f"  Content length: {len(response.text)} characters")
        print(f"  Content type: {response.headers.get('content-type', 'unknown')}")
        
        if response.status_code == 200:
            print(f"  First 200 characters: {response.text[:200]}")
            
            # Step 5: Parse with pandas
            print(f"Step 5 - Pandas parsing...")
            
            import pandas as pd
            import io
            
            df = pd.read_csv(io.StringIO(response.text), skipinitialspace=True)
            print(f"  ✅ Pandas parsing successful!")
            print(f"  Shape: {df.shape}")
            print(f"  Columns: {list(df.columns)}")
            
            return True
            
        else:
            print(f"  ❌ HTTP request failed")
            return False
            
    except Exception as e:
        print(f"❌ Step-by-step test failed: {str(e)}")
        traceback.print_exc()
        return False

def main():
    """Test the exact parameters that caused the error."""
    print("🔧 Testing Exact Parameters Fix")
    print("=" * 40)
    
    success1 = test_step_by_step()
    
    if success1:
        print(f"\n" + "="*40)
        success2 = test_exact_parameters()
        
        if success2:
            print(f"\n🎉 Fix successful!")
            print(f"✅ The dashboard should now work with these parameters")
        else:
            print(f"\n❌ Extractor still has issues")
    else:
        print(f"\n❌ Basic components still failing")

if __name__ == "__main__":
    main()