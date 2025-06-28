#!/usr/bin/env python3
"""
Ocean Data Pipeline Orchestrator

This script coordinates the extraction, transformation, and loading of ocean data.
It provides error handling, logging, and data validation between pipeline steps.

Usage:
    python run_pipeline.py                    # Run full pipeline
    python run_pipeline.py --step extract     # Run only extraction
    python run_pipeline.py --step transform   # Run only transformation  
    python run_pipeline.py --step load        # Run only loading
    python run_pipeline.py --validate-only    # Just validate data quality
"""

import argparse
import logging
import sys
import traceback
from pathlib import Path
from datetime import datetime
import pandas as pd
import duckdb

# Import pipeline modules
sys.path.append(str(Path(__file__).parent))
from pipeline.extract import download_sea_surface_data
from pipeline.transform import run as transform_run
from pipeline.load import load_to_duckdb

# Setup logging
def setup_logging():
    """Configure logging for the pipeline."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"pipeline_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

class DataValidator:
    """Validates data quality at different pipeline stages."""
    
    @staticmethod
    def validate_raw_data(file_path: Path) -> dict:
        """Validate raw extracted data."""
        logger = logging.getLogger(__name__)
        logger.info(f"🔍 Validating raw data: {file_path}")
        
        if not file_path.exists():
            raise FileNotFoundError(f"Raw data file not found: {file_path}")
        
        df = pd.read_csv(file_path)
        
        validation_results = {
            "file_exists": True,
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns": list(df.columns),
            "has_data": len(df) > 0,
            "file_size_mb": file_path.stat().st_size / (1024 * 1024)
        }
        
        logger.info(f"✅ Raw data validation: {validation_results['row_count']} rows, {validation_results['column_count']} columns")
        return validation_results
    
    @staticmethod
    def validate_clean_data(file_path: Path) -> dict:
        """Validate cleaned/transformed data."""
        logger = logging.getLogger(__name__)
        logger.info(f"🔍 Validating clean data: {file_path}")
        
        if not file_path.exists():
            raise FileNotFoundError(f"Clean data file not found: {file_path}")
        
        df = pd.read_csv(file_path)
        
        # Check for required columns
        required_columns = ["time", "temperature", "salinity"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        # Check data quality
        null_counts = df.isnull().sum()
        
        validation_results = {
            "file_exists": True,
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns": list(df.columns),
            "missing_required_columns": missing_columns,
            "null_counts": null_counts.to_dict(),
            "has_valid_timestamps": df['time'].notna().sum() if 'time' in df.columns else 0,
            "temperature_range": [df['temperature'].min(), df['temperature'].max()] if 'temperature' in df.columns else None,
            "salinity_range": [df['salinity'].min(), df['salinity'].max()] if 'salinity' in df.columns else None
        }
        
        # Quality checks
        if missing_columns:
            logger.warning(f"⚠️ Missing required columns: {missing_columns}")
        
        if validation_results["null_counts"]:
            logger.info(f"📊 Null value counts: {validation_results['null_counts']}")
        
        logger.info(f"✅ Clean data validation: {validation_results['row_count']} rows validated")
        return validation_results
    
    @staticmethod
    def validate_database(db_path: Path) -> dict:
        """Validate data in the database."""
        logger = logging.getLogger(__name__)
        logger.info(f"🔍 Validating database: {db_path}")
        
        if not db_path.exists():
            raise FileNotFoundError(f"Database file not found: {db_path}")
        
        conn = duckdb.connect(str(db_path))
        
        try:
            # Check if table exists
            tables = conn.execute("SHOW TABLES").fetchall()
            table_names = [table[0] for table in tables]
            
            if 'sea_surface' not in table_names:
                raise ValueError("sea_surface table not found in database")
            
            # Get table info
            row_count = conn.execute("SELECT COUNT(*) FROM sea_surface").fetchone()[0]
            columns = conn.execute("DESCRIBE sea_surface").fetchall()
            column_names = [col[0] for col in columns]
            
            # Sample data check
            sample_data = conn.execute("SELECT * FROM sea_surface LIMIT 5").fetchall()
            
            validation_results = {
                "database_exists": True,
                "tables": table_names,
                "row_count": row_count,
                "columns": column_names,
                "sample_data_available": len(sample_data) > 0,
                "file_size_mb": db_path.stat().st_size / (1024 * 1024)
            }
            
            logger.info(f"✅ Database validation: {row_count} rows in sea_surface table")
            return validation_results
            
        finally:
            conn.close()

class PipelineOrchestrator:
    """Main pipeline orchestrator class."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.validator = DataValidator()
        
        # Define file paths
        self.raw_data_path = Path("data") / "sea_surface_sample.csv"
        self.clean_data_path = Path("data") / "clean" / "sea_surface_clean.csv"
        self.db_path = Path("data") / "ocean_data.duckdb"
    
    def run_extract(self) -> bool:
        """Run the data extraction step."""
        try:
            self.logger.info("🚀 Starting EXTRACT phase...")
            download_sea_surface_data()
            
            # Validate extracted data
            validation = self.validator.validate_raw_data(self.raw_data_path)
            if not validation["has_data"]:
                raise ValueError("No data was extracted")
            
            self.logger.info("✅ EXTRACT phase completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ EXTRACT phase failed: {str(e)}")
            self.logger.debug(traceback.format_exc())
            return False
    
    def run_transform(self) -> bool:
        """Run the data transformation step."""
        try:
            self.logger.info("🔄 Starting TRANSFORM phase...")
            
            # Check if raw data exists
            if not self.raw_data_path.exists():
                raise FileNotFoundError("Raw data not found. Run extract step first.")
            
            transform_run()
            
            # Validate transformed data
            validation = self.validator.validate_clean_data(self.clean_data_path)
            if validation["missing_required_columns"]:
                raise ValueError(f"Required columns missing: {validation['missing_required_columns']}")
            
            self.logger.info("✅ TRANSFORM phase completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ TRANSFORM phase failed: {str(e)}")
            self.logger.debug(traceback.format_exc())
            return False
    
    def run_load(self) -> bool:
        """Run the data loading step."""
        try:
            self.logger.info("💾 Starting LOAD phase...")
            
            # Check if clean data exists
            if not self.clean_data_path.exists():
                raise FileNotFoundError("Clean data not found. Run transform step first.")
            
            load_to_duckdb()
            
            # Validate database
            validation = self.validator.validate_database(self.db_path)
            if validation["row_count"] == 0:
                raise ValueError("No data was loaded into database")
            
            self.logger.info("✅ LOAD phase completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ LOAD phase failed: {str(e)}")
            self.logger.debug(traceback.format_exc())
            return False
    
    def validate_pipeline(self) -> dict:
        """Validate the entire pipeline data quality."""
        self.logger.info("🔍 Running full pipeline validation...")
        
        results = {}
        
        try:
            if self.raw_data_path.exists():
                results["raw_data"] = self.validator.validate_raw_data(self.raw_data_path)
            else:
                results["raw_data"] = {"error": "Raw data file not found"}
        except Exception as e:
            results["raw_data"] = {"error": str(e)}
        
        try:
            if self.clean_data_path.exists():
                results["clean_data"] = self.validator.validate_clean_data(self.clean_data_path)
            else:
                results["clean_data"] = {"error": "Clean data file not found"}
        except Exception as e:
            results["clean_data"] = {"error": str(e)}
        
        try:
            if self.db_path.exists():
                results["database"] = self.validator.validate_database(self.db_path)
            else:
                results["database"] = {"error": "Database file not found"}
        except Exception as e:
            results["database"] = {"error": str(e)}
        
        self.logger.info("✅ Pipeline validation completed")
        return results
    
    def run_full_pipeline(self) -> bool:
        """Run the complete ETL pipeline."""
        self.logger.info("🚀 Starting FULL PIPELINE execution...")
        start_time = datetime.now()
        
        success = True
        
        # Run each phase
        if not self.run_extract():
            success = False
        elif not self.run_transform():
            success = False
        elif not self.run_load():
            success = False
        
        # Final validation
        if success:
            validation_results = self.validate_pipeline()
            self.logger.info("📊 Final validation results:")
            for stage, results in validation_results.items():
                if "error" in results:
                    self.logger.error(f"  {stage}: {results['error']}")
                    success = False
                else:
                    self.logger.info(f"  {stage}: ✅ Valid")
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        if success:
            self.logger.info(f"🎉 FULL PIPELINE completed successfully in {duration}")
        else:
            self.logger.error(f"❌ FULL PIPELINE failed after {duration}")
        
        return success

def main():
    """Main entry point for the pipeline orchestrator."""
    parser = argparse.ArgumentParser(description="Ocean Data Pipeline Orchestrator")
    parser.add_argument("--step", choices=["extract", "transform", "load"], 
                       help="Run only a specific pipeline step")
    parser.add_argument("--validate-only", action="store_true", 
                       help="Only run data validation checks")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("=" * 60)
    logger.info("🌊 OCEAN DATA PIPELINE ORCHESTRATOR")
    logger.info("=" * 60)
    
    # Initialize orchestrator
    orchestrator = PipelineOrchestrator()
    
    try:
        if args.validate_only:
            # Just run validation
            results = orchestrator.validate_pipeline()
            print("\n📊 VALIDATION RESULTS:")
            print("-" * 40)
            for stage, result in results.items():
                if "error" in result:
                    print(f"{stage.upper()}: ❌ {result['error']}")
                else:
                    print(f"{stage.upper()}: ✅ Valid")
                    if "row_count" in result:
                        print(f"  - Rows: {result['row_count']}")
                    if "columns" in result:
                        print(f"  - Columns: {len(result['columns'])}")
            return
        
        success = False
        
        if args.step == "extract":
            success = orchestrator.run_extract()
        elif args.step == "transform":
            success = orchestrator.run_transform() 
        elif args.step == "load":
            success = orchestrator.run_load()
        else:
            # Run full pipeline
            success = orchestrator.run_full_pipeline()
        
        if success:
            logger.info("🎉 Pipeline execution completed successfully!")
            print("\n✅ SUCCESS: Pipeline completed successfully!")
            print("📁 You can now run: streamlit run dashboard/app.py")
        else:
            logger.error("❌ Pipeline execution failed!")
            print("\n❌ FAILURE: Pipeline execution failed!")
            print("📋 Check the logs for detailed error information.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("⏹️ Pipeline execution interrupted by user")
        print("\n⏹️ Pipeline execution was interrupted")
    except Exception as e:
        logger.error(f"💥 Unexpected error: {str(e)}")
        logger.debug(traceback.format_exc())
        print(f"\n💥 Unexpected error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()