#!/usr/bin/env python3
"""
Phase 2 Setup and Testing Script

This script helps set up and test the Phase 2 interactive dashboard
with dynamic API integration.
"""

import sys
import subprocess
import logging
from pathlib import Path
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if all required components are available."""
    logger.info("🔍 Checking Phase 2 dependencies...")
    
    # Check if Phase 1 components exist
    phase1_files = [
        'config.py',
        'cache_manager.py',
        'pipeline/extract.py'
    ]
    
    missing_files = []
    for file_path in phase1_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        logger.error(f"❌ Missing Phase 1 files: {missing_files}")
        logger.error("Please integrate Phase 1 components first!")
        return False
    
    # Check if we can import Phase 1 components
    try:
        from config import get_coverage_bounds, coordinate_converter
        from cache_manager import cache_manager
        from pipeline.extract import ERDDAPExtractor
        logger.info("✅ Phase 1 components imported successfully")
        return True
    except ImportError as e:
        logger.error(f"❌ Cannot import Phase 1 components: {e}")
        return False

def test_api_connection():
    """Test API connection with dynamic extractor."""
    logger.info("🧪 Testing API connection...")
    
    try:
        from pipeline.extract import ERDDAPExtractor
        
        extractor = ERDDAPExtractor()
        success, message = extractor.test_api_connection()
        
        if success:
            logger.info(f"✅ API connection test passed: {message}")
            return True
        else:
            logger.error(f"❌ API connection test failed: {message}")
            return False
            
    except Exception as e:
        logger.error(f"❌ API connection test error: {e}")
        return False

def test_coordinate_validation():
    """Test coordinate validation for dashboard."""
    logger.info("🧪 Testing coordinate validation...")
    
    try:
        from config import coordinate_converter
        
        # Test valid coordinates
        test_coords = [
            (40.0, -30.0, True),   # Valid North Atlantic
            (60.0, -20.0, True),   # Valid North Atlantic
            (10.0, -30.0, False),  # Too far south
            (40.0, -100.0, False), # Too far west
        ]
        
        all_passed = True
        for lat, lon, expected_valid in test_coords:
            is_valid, message = coordinate_converter.validate_coordinates(lat, lon)
            
            if is_valid == expected_valid:
                logger.info(f"✅ {lat:.1f}°N, {lon:.1f}°W: {message}")
            else:
                logger.error(f"❌ {lat:.1f}°N, {lon:.1f}°W: Expected {expected_valid}, got {is_valid}")
                all_passed = False
        
        return all_passed
        
    except Exception as e:
        logger.error(f"❌ Coordinate validation test error: {e}")
        return False

def test_cache_system():
    """Test cache system for dashboard."""
    logger.info("🧪 Testing cache system...")
    
    try:
        from cache_manager import cache_manager
        
        # Get cache stats
        stats = cache_manager.get_cache_stats()
        logger.info(f"✅ Cache system active: {stats['active_entries']} entries")
        
        # Test cache cleanup
        cleaned = cache_manager.cleanup_expired_cache()
        logger.info(f"✅ Cache cleanup: {cleaned} expired entries removed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Cache system test error: {e}")
        return False

def create_sample_data():
    """Create sample data for testing dashboard."""
    logger.info("🧪 Creating sample data for testing...")
    
    try:
        from pipeline.extract import ERDDAPExtractor
        
        extractor = ERDDAPExtractor()
        
        # Fetch sample data
        logger.info("Fetching sample data (this may take a moment)...")
        df, metadata = extractor.fetch_data_for_location(
            latitude=40.0,
            longitude=-30.0,
            start_date="1980-01-01",
            end_date="1982-12-01"
        )
        
        if not df.empty:
            logger.info(f"✅ Sample data created: {len(df)} records")
            logger.info(f"   Temperature range: {df['temperature'].min():.1f}°C - {df['temperature'].max():.1f}°C")
            logger.info(f"   Salinity range: {df['salinity'].min():.1f} - {df['salinity'].max():.1f} PSU")
            return True
        else:
            logger.error("❌ No sample data retrieved")
            return False
            
    except Exception as e:
        logger.error(f"❌ Sample data creation error: {e}")
        return False

def launch_dashboard():
    """Launch the enhanced dashboard."""
    logger.info("🚀 Launching enhanced dashboard...")
    
    try:
        # Check if streamlit is available
        result = subprocess.run(['streamlit', '--version'], 
                              capture_output=True, text=True)
        if result.returncode != 0:
            logger.error("❌ Streamlit not found. Please install: pip install streamlit")
            return False
        
        logger.info("✅ Streamlit found, launching dashboard...")
        logger.info("🌊 Dashboard will open in your browser...")
        logger.info("📍 Try clicking on different locations on the map!")
        
        # Launch dashboard
        subprocess.run(['streamlit', 'run', 'dashboard/app.py'])
        
    except KeyboardInterrupt:
        logger.info("⏹️ Dashboard stopped by user")
    except Exception as e:
        logger.error(f"❌ Dashboard launch error: {e}")
        return False

def run_comprehensive_test():
    """Run all Phase 2 tests."""
    logger.info("🚀 Running comprehensive Phase 2 tests...")
    logger.info("=" * 60)
    
    test_results = []
    
    # Test 1: Dependencies
    test_results.append(("Dependencies", check_dependencies()))
    
    # Test 2: API Connection
    test_results.append(("API Connection", test_api_connection()))
    
    # Test 3: Coordinate Validation
    test_results.append(("Coordinate Validation", test_coordinate_validation()))
    
    # Test 4: Cache System
    test_results.append(("Cache System", test_cache_system()))
    
    # Test 5: Sample Data
    test_results.append(("Sample Data", create_sample_data()))
    
    # Report results
    logger.info("\n📊 TEST RESULTS:")
    logger.info("-" * 40)
    
    all_passed = True
    for test_name, passed in test_results:
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"{test_name}: {status}")
        if not passed:
            all_passed = False
    
    logger.info("-" * 40)
    
    if all_passed:
        logger.info("🎉 All Phase 2 tests passed!")
        logger.info("✅ Enhanced dashboard is ready to use")
        return True
    else:
        logger.error("❌ Some Phase 2 tests failed")
        logger.error("Please fix the issues before using the dashboard")
        return False

def main():
    """Main testing function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Phase 2 Setup and Testing")
    parser.add_argument("--test", action="store_true", help="Run comprehensive tests")
    parser.add_argument("--launch", action="store_true", help="Launch dashboard")
    parser.add_argument("--check-deps", action="store_true", help="Check dependencies only")
    parser.add_argument("--sample-data", action="store_true", help="Create sample data")
    
    args = parser.parse_args()
    
    if args.check_deps:
        success = check_dependencies()
        sys.exit(0 if success else 1)
    
    elif args.sample_data:
        success = create_sample_data()
        sys.exit(0 if success else 1)
    
    elif args.test:
        success = run_comprehensive_test()
        sys.exit(0 if success else 1)
    
    elif args.launch:
        if check_dependencies():
            launch_dashboard()
        else:
            logger.error("❌ Cannot launch dashboard - dependencies not met")
            sys.exit(1)
    
    else:
        # Default: run tests then launch if successful
        logger.info("🌊 Phase 2 Setup: Testing then launching dashboard...")
        
        if run_comprehensive_test():
            time.sleep(2)  # Brief pause
            logger.info("\n🚀 Launching dashboard...")
            launch_dashboard()
        else:
            logger.error("❌ Tests failed - not launching dashboard")
            sys.exit(1)

if __name__ == "__main__":
    main()