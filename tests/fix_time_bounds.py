#!/usr/bin/env python3
"""
Fix the time bounds issue by finding the actual valid time range
"""

import requests
from datetime import datetime

def test_time_indices():
    """Test different time indices to find the valid range."""
    
    print("🔍 Testing time indices to find valid range...")
    
    # Use known good spatial coordinates
    base_url = "https://erddap.ifremer.fr/erddap/griddap/SDC_NAT_CLIM_TS_V1_025_m.csv"
    
    # Test different time indices
    test_times = [
        0,     # Original working (1955)
        50,    # ~1959
        100,   # ~1963
        200,   # ~1971
        300,   # ~1980 (currently failing)
        400,   # ~1988
        500,   # ~1996
        600,   # ~2005
        700,   # ~2013
        731,   # Should be last index (2015)
        732,   # Should be invalid (beyond end)
    ]
    
    valid_times = []
    
    for time_idx in test_times:
        url = f"{base_url}?Temperature[{time_idx}][106][90][60]"
        
        try:
            response = requests.head(url, timeout=5)
            status = response.status_code
            
            if status == 200:
                print(f"  Time {time_idx}: ✅ Valid")
                valid_times.append(time_idx)
            else:
                print(f"  Time {time_idx}: ❌ Invalid ({status})")
                
        except Exception as e:
            print(f"  Time {time_idx}: ❌ Error ({e})")
    
    if valid_times:
        print(f"\n✅ Valid time indices: {min(valid_times)} to {max(valid_times)}")
        return min(valid_times), max(valid_times)
    else:
        print("\n❌ No valid time indices found")
        return None, None

def test_time_ranges():
    """Test time ranges to see what works."""
    
    print(f"\n🔍 Testing time ranges...")
    
    base_url = "https://erddap.ifremer.fr/erddap/griddap/SDC_NAT_CLIM_TS_V1_025_m.csv"
    
    # Test small ranges
    test_ranges = [
        (0, 1),     # Original range
        (0, 11),    # First year  
        (0, 59),    # First 5 years
        (100, 111), # Mid-range year
        (200, 211), # Later year
    ]
    
    for start, end in test_ranges:
        url = f"{base_url}?Temperature[{start}:{end}][106][90][60]"
        
        try:
            response = requests.head(url, timeout=5)
            status = response.status_code
            
            if status == 200:
                print(f"  Range {start}:{end}: ✅ Valid")
            else:
                print(f"  Range {start}:{end}: ❌ Invalid ({status})")
                
        except Exception as e:
            print(f"  Range {start}:{end}: ❌ Error")

def find_dataset_time_bounds():
    """Try to find the actual time bounds of the dataset."""
    
    print(f"\n🔍 Finding actual dataset time bounds...")
    
    # Binary search to find the maximum valid time index
    base_url = "https://erddap.ifremer.fr/erddap/griddap/SDC_NAT_CLIM_TS_V1_025_m.csv"
    
    def is_valid_time(time_idx):
        url = f"{base_url}?Temperature[{time_idx}][106][90][60]"
        try:
            response = requests.head(url, timeout=3)
            return response.status_code == 200
        except:
            return False
    
    # Find maximum valid time index
    low, high = 0, 1000
    max_valid = 0
    
    while low <= high:
        mid = (low + high) // 2
        if is_valid_time(mid):
            max_valid = mid
            low = mid + 1
        else:
            high = mid - 1
    
    print(f"✅ Maximum valid time index: {max_valid}")
    
    # Convert to date
    from config import coordinate_converter
    try:
        max_date = coordinate_converter.time_index_to_date(max_valid)
        print(f"✅ Maximum date: {max_date}")
    except:
        print("❌ Could not convert time index to date")
    
    return max_valid

def fix_time_conversion():
    """Fix the time conversion to use the actual dataset bounds."""
    
    max_time_idx = find_dataset_time_bounds()
    
    if max_time_idx:
        print(f"\n🔧 Recommended fixes:")
        print(f"1. Update GRID_CONFIG['time']['grid_size'] to {max_time_idx + 1}")
        print(f"2. Limit date ranges to prevent out-of-bounds access")
        print(f"3. Add validation in date_to_time_index()")
        
        # Calculate what this means for date range
        years = max_time_idx / 12
        end_year = 1955 + years
        print(f"4. Dataset appears to cover ~{years:.1f} years (1955 to ~{end_year:.0f})")

def test_working_dates():
    """Test with dates that should definitely work."""
    
    print(f"\n🧪 Testing with safe date ranges...")
    
    from config import coordinate_converter, query_builder
    
    # Test very conservative date ranges
    test_dates = [
        ("1955-01-01", "1955-12-01"),  # First year
        ("1960-01-01", "1960-12-01"),  # Known working year
        ("1970-01-01", "1970-12-01"),  # Mid-range
    ]
    
    for start_date, end_date in test_dates:
        start_idx = coordinate_converter.date_to_time_index(start_date)
        end_idx = coordinate_converter.date_to_time_index(end_date)
        
        print(f"  {start_date} to {end_date}: indices {start_idx}:{end_idx}")
        
        # Test the URL
        url = query_builder.build_query_url(32.5, -70.0, start_date, end_date)
        
        try:
            response = requests.head(url, timeout=5)
            status = "✅" if response.status_code == 200 else "❌"
            print(f"    Status: {response.status_code} {status}")
        except Exception as e:
            print(f"    Error: {e}")

def main():
    """Run all time bounds tests."""
    print("🔧 Fixing Time Bounds Issue")
    print("=" * 40)
    
    test_time_indices()
    test_time_ranges()
    fix_time_conversion()
    test_working_dates()
    
    print(f"\n🎯 The issue is likely that time indices 300+ are out of bounds")
    print(f"💡 Try using earlier dates (1955-1970) until we fix the time bounds")

if __name__ == "__main__":
    main()