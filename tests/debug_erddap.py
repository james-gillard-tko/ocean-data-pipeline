#!/usr/bin/env python3
"""
Debug script to test ERDDAP URL generation and fix the 400 error
"""

import requests
from config import query_builder, coordinate_converter

def test_coordinate_conversion():
    """Test coordinate conversion with the coordinates that failed."""
    
    # The coordinates that failed (based on your error URL)
    test_lat, test_lon = 40.0, -30.0  # Example coordinates
    
    print(f"Testing coordinates: {test_lat}°N, {test_lon}°W")
    
    # Convert to grid indices
    lat_idx = coordinate_converter.lat_to_grid_index(test_lat)
    lon_idx = coordinate_converter.lon_to_grid_index(test_lon)
    
    print(f"Grid indices: lat={lat_idx}, lon={lon_idx}")
    
    # Convert back to coordinates
    actual_lat = coordinate_converter.grid_index_to_lat(lat_idx)
    actual_lon = coordinate_converter.grid_index_to_lon(lon_idx)
    
    print(f"Actual coordinates: {actual_lat:.3f}°N, {actual_lon:.3f}°W")
    
    return lat_idx, lon_idx

def test_erddap_url():
    """Test ERDDAP URL generation."""
    
    # Test with known working coordinates (from original pipeline)
    lat, lon = 32.5, -70.0
    start_date, end_date = "1960-01-01", "1960-02-01"
    
    print(f"\nTesting ERDDAP URL generation:")
    print(f"Coordinates: {lat}°N, {lon}°W")
    print(f"Date range: {start_date} to {end_date}")
    
    # Build URL
    url = query_builder.build_query_url(lat, lon, start_date, end_date)
    print(f"Generated URL: {url}")
    
    # Test the URL
    print(f"\nTesting URL...")
    try:
        response = requests.head(url, timeout=10)  # Use HEAD to avoid downloading data
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            print("✅ URL is valid!")
        else:
            print(f"❌ URL failed with status {response.status_code}")
    except Exception as e:
        print(f"❌ URL test failed: {e}")

def test_simple_erddap_query():
    """Test with the simplest possible ERDDAP query."""
    
    # Use the exact same parameters as the original working pipeline
    base_url = "https://erddap.ifremer.fr/erddap/griddap/SDC_NAT_CLIM_TS_V1_025_m.csv"
    
    # Original working query
    params = "Temperature%5B0%5D%5B106%5D%5B90%5D%5B60%5D%2CSalinity%5B0%5D%5B106%5D%5B90%5D%5B60%5D"
    url = f"{base_url}?{params}"
    
    print(f"\nTesting original working URL:")
    print(f"URL: {url}")
    
    try:
        response = requests.head(url, timeout=10)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            print("✅ Original URL still works!")
        else:
            print(f"❌ Original URL failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Original URL test failed: {e}")

def debug_failed_url():
    """Debug the specific URL that failed."""
    
    # Your failed URL (reconstructed from the error)
    failed_url = "https://erddap.ifremer.fr/erddap/griddap/SDC_NAT_CLIM_TS_V1_025_m.csv?Temperature%5B300:371%5D%5B106%5D%5B160%5D%5B199%5D,Salinity%5B300:371%5D%5B106%5D%5B160%5D%5B199%5D"
    
    print(f"\nDebugging failed URL:")
    print(f"URL: {failed_url}")
    
    # Decode the parameters
    print(f"\nDecoded parameters:")
    print(f"Temperature[300:371][106][160][199]")
    print(f"Salinity[300:371][106][160][199]")
    print(f"")
    print(f"Time indices: 300 to 371 (probably too high)")
    print(f"Depth index: 106 (surface - correct)")
    print(f"Lat index: 160 (needs checking)")
    print(f"Lon index: 199 (needs checking)")
    
    # Check if time indices are valid
    time_idx_300 = coordinate_converter.time_index_to_date(300)
    time_idx_371 = coordinate_converter.time_index_to_date(371)
    
    print(f"\nTime index 300 = {time_idx_300}")
    print(f"Time index 371 = {time_idx_371}")
    
    # Check dataset bounds
    from config import GRID_CONFIG
    print(f"\nDataset bounds:")
    print(f"Time grid size: {GRID_CONFIG['time']['grid_size']}")
    print(f"Lat grid size: {GRID_CONFIG['latitude']['grid_size']}")
    print(f"Lon grid size: {GRID_CONFIG['longitude']['grid_size']}")

def fix_coordinate_bounds():
    """Check and potentially fix coordinate bounds."""
    
    from config import GRID_CONFIG
    
    print(f"\nChecking grid configuration:")
    print(f"Latitude: {GRID_CONFIG['latitude']['min']}° to {GRID_CONFIG['latitude']['max']}°")
    print(f"Longitude: {GRID_CONFIG['longitude']['min']}° to {GRID_CONFIG['longitude']['max']}°")
    print(f"Time: {GRID_CONFIG['time']['start']} to {GRID_CONFIG['time']['end']}")
    
    # Test a coordinate that should definitely work
    test_coords = [
        (40.0, -30.0),  # Mid Atlantic
        (50.0, -20.0),  # North Atlantic
        (32.5, -70.0),  # Original working coordinate
    ]
    
    for lat, lon in test_coords:
        lat_idx = coordinate_converter.lat_to_grid_index(lat)
        lon_idx = coordinate_converter.lon_to_grid_index(lon)
        
        print(f"\n{lat}°N, {lon}°W → grid[{lat_idx}, {lon_idx}]")
        
        # Check if indices are within bounds
        lat_ok = 0 <= lat_idx < GRID_CONFIG['latitude']['grid_size']
        lon_ok = 0 <= lon_idx < GRID_CONFIG['longitude']['grid_size']
        
        print(f"  Lat index valid: {lat_ok}")
        print(f"  Lon index valid: {lon_ok}")

def main():
    """Run all debugging tests."""
    print("🔍 ERDDAP URL Debugging")
    print("=" * 50)
    
    try:
        test_coordinate_conversion()
        test_simple_erddap_query()
        debug_failed_url()
        fix_coordinate_bounds()
        test_erddap_url()
        
        print(f"\n🎯 RECOMMENDATIONS:")
        print(f"1. Check if time indices are within bounds (0-731)")
        print(f"2. Verify coordinate conversion is correct")
        print(f"3. Test with original working coordinates first")
        print(f"4. Check dataset metadata on ERDDAP server")
        
    except Exception as e:
        print(f"❌ Debug script failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()