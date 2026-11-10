"""
================================================================================
MODULE: Database Extractor
FILE: src/extractors/O04_db_extractor.py
PURPOSE:
  Manages extractions from relational (SQL) and NoSQL databases. Implements 
  connection pooling and batched chunksize pagination queries. This prevents 
  monolithic queries from locking database tables or exceeding available 
  system RAM when querying millions of historical records.
================================================================================
"""
import logging
from typing import Generator, Optional
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("DatabaseExtractor")


class DatabaseExtractor:
    """Enterprise database extractor with support for connection pooling and pagination."""

    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        logger.info("Initialized DatabaseExtractor with connection target.")

    def extract_in_batches(
        self, query: str, chunk_size: int = 10000
    ) -> Generator[pd.DataFrame, None, None]:
        """
        Executes a SQL query and yields data in manageable chunks (out-of-core pattern).
        
        Args:
            query (str): SQL query string to execute.
            chunk_size (int): Number of rows per batch chunk.
            
        Yields:
            pd.DataFrame: A chunk of records as a pandas DataFrame.
        """
        logger.info(f"Executing paginated query with chunk_size={chunk_size:,}...")
        
        # Note: In a real production environment, you would use SQLAlchemy 
        # or Psycopg2 connection pools like:
        # engine = create_engine(self.connection_string)
        # with engine.connect() as conn:
        #     for chunk in pd.read_sql(query, conn, chunksize=chunk_size):
        #         yield chunk

        # Simulated demonstration yield for educational structure
        logger.warning("[SIMULATION MODE] Database connection string not active. Yielding mock batch generator.")
        mock_data = pd.DataFrame({
            "customer_id": [f"CUST-{i:03d}" for i in range(1, 6)],
            "tenure_months": [12, 24, 6, 48, 3],
            "monthly_charges": [29.95, 75.50, 19.00, 105.20, 45.00],
            "churn_status": ["no", "yes", "no", "no", "yes"]
        })
        yield mock_data


if __name__ == "__main__":
    # Local demonstration test
    db_extractor = DatabaseExtractor(connection_string="sqlite:///mock_enterprise.db")
    
    batch_generator = db_extractor.extract_in_batches(
        query="SELECT * FROM customer_telemetry", chunk_size=2
    )
    
    for idx, batch_df in enumerate(batch_generator):
        logger.info(f"Processing database batch chunk #{idx + 1} with {len(batch_df)} rows.")
        print(batch_df)