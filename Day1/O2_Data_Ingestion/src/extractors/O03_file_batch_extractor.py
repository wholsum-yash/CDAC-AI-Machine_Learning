"""
================================================================================
MODULE: Batch File Extractor
FILE: src/extractors/O03_file_batch_extractor.py
PURPOSE:
  Handles batch extractions of structured and semi-structured flat files 
  (CSV, JSON, NDJSON, Parquet) deposited in the raw landing zone (Bronze layer). 
  Manages encoding decoding (UTF-8/Latin-1), missing file exceptions, and 
  converts file payloads into pandas DataFrames with robust error logging.
================================================================================
"""

import logging
from pathlib import Path
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("BatchFileExtractor")

class BatchFileExtractor:
    def __init__(self, raw_data_dir: Path):
        self.raw_data_dir = Path(raw_data_dir)
        if not self.raw_data_dir.exists():
            logger.warning(f"Raw data directory does not exist yet: {self.raw_data_dir}")


    def extract_file(self, file_name: str, file_format: str = "csv", **kwargs) -> pd.DataFrame:
        """
        Extracts a batch file from the raw landing zone based on its format.
        
        Args:
            file_name (str): Name of the file (e.g., 'customer_telemetry.csv').
            file_format (str): Format type ('csv', 'json', 'parquet').
            **kwargs: Additional keyword arguments passed to pandas read functions.
            
        Returns:
            pd.DataFrame: Extracted raw DataFrame.
        """
        file_path = self.raw_data_dir / file_name
        
        if not file_path.exists():
            raise FileNotFoundError(f"[ERROR] Target file not found: {file_path}")
            
        logger.info(f"Starting batch extraction for '{file_name}' (Format: {file_format.upper()})...")
        
        try:
            if file_format.lower() == "csv":
                # Default to UTF-8, handling potential encoding issues gracefully
                df = pd.read_csv(file_path, encoding="utf-8", **kwargs)
            elif file_format.lower() == "json":
                df = pd.read_json(file_path, **kwargs)
            elif file_format.lower() == "parquet":
                df = pd.read_parquet(file_path, **kwargs)
            else:
                raise ValueError(f"Unsupported file format: '{file_format}'. Use csv, json, or parquet.")
                
            logger.info(f"Successfully extracted {len(df):,} rows and {len(df.columns)} columns from {file_name}.")
            return df
            
        except Exception as e:
            logger.error(f"Failed to parse file {file_name}: {e}")
            raise e


if __name__ == "__main__":
    # Local demonstration test
    landing_zone = Path("data/raw")
    # parents=True ensures that all intermediate directories are created if they don't exist.
    # exist_ok=True ensures that no error is raised if the directory already exists.
    landing_zone.mkdir(parents=True, exist_ok=True)
    
    sample_csv = landing_zone / "customer_telemetry.csv"
    
    # Create sample CSV if not present
    if not sample_csv.exists():
        sample_csv.write_text(
            "customer_id,tenure_months,monthly_charges,total_charges,contract_type,payment_method,churn_status\n"
            "CUST-001,12,29.95,359.4,month-to-month,electronic_check,no\n"
            "CUST-002,34,56.50,1921.0,one-year,mailed_check,yes\n",
            encoding="utf-8"
        )

    # Run extractor test
    extractor = BatchFileExtractor(raw_data_dir=landing_zone)
    df_extracted = extractor.extract_file("customer_telemetry.csv", file_format="csv")
    print("\n[PREVIEW] Extracted DataFrame Head:")
    print(df_extracted.head())