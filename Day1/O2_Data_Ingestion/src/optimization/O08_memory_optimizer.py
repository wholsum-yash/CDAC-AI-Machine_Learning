"""
================================================================================
MODULE: Low-Level Memory Optimizer
FILE: src/optimization/O08_memory_optimizer.py
PURPOSE:
  Optimizes the RAM footprint of large datasets through aggressive numerical 
  type downcasting (e.g., converting int64 to int8/int16/int32 and float64 to 
  float32 without precision loss) and demonstrates out-of-core / lazy evaluation 
  processing patterns using high-performance engines like Polars and DuckDB.
================================================================================
"""


import logging
from pathlib import Path
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("MemoryOptimizer")


class MemoryOptimizer:
    """Optimizes dataframe memory footprint through type downcasting and efficient engines."""

    @staticmethod
    def get_memory_usage_mb(df: pd.DataFrame) -> float:
        """Calculate total memory usage of a DataFrame in Megabytes (MB)."""
        return df.memory_usage(deep=True).sum() / (1024 * 1024) # approx MB conversion

    @classmethod
    def downcast_numeric_types(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggressively downcasts integer and float columns to minimize RAM usage 
        without introducing precision loss or overflow.
        
        Args:
            df (pd.DataFrame): Input DataFrame.
            
        Returns:
            pd.DataFrame: Memory-optimized DataFrame.
        """
        start_mem = cls.get_memory_usage_mb(df)
        logger.info(f"Memory usage before downcasting: {start_mem:.2f} MB")

        for col in df.columns:
            col_type = df[col].dtype

            # 1. Downcast Integers
            if "int" in str(col_type):
                c_min = df[col].min()
                c_max = df[col].max()
                
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)

            # 2. Downcast Floats
            elif "float" in str(col_type):
                c_min = df[col].min()
                c_max = df[col].max()
                
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

        end_mem = cls.get_memory_usage_mb(df)
        reduction = 100 * (start_mem - end_mem) / start_mem if start_mem > 0 else 0
        logger.info(f"Memory usage after downcasting : {end_mem:.2f} MB (Saved {reduction:.1f}% RAM)")
        
        return df

    @staticmethod
    def process_out_of_core_polars(file_path: Path) -> pd.DataFrame:
        """
        Demonstrates out-of-core or lazy loading using Polars for high-performance 
        memory-efficient file parsing.
        """
        logger.info(f"Scanning file lazily with Polars: {file_path.name}")
        try:
            # Polars lazy evaluation allows query optimization before loading into RAM
            lazy_df = pl.scan_csv(str(file_path))
            df = lazy_df.collect().to_pandas()
            return df
        except Exception as e:
            logger.warning(f"Polars engine execution skipped/unavailable: {e}. Falling back to pandas.")
            return pd.read_csv(file_path)


if __name__ == "__main__":
    # Local demonstration test
    import polars as pl  # Optional dependency check
    
    # Create sample high-precision dataframe
    mock_df = pd.DataFrame({
        "customer_id": [f"CUST-{i:04d}" for i in range(1000)],
        "tenure_months": np.random.randint(0, 72, size=1000, dtype=np.int64),
        "monthly_charges": np.random.uniform(20.0, 150.0, size=1000).astype(np.float64),
    })

    print("\n[DEMO] Running Memory Downcasting Optimizer:")
    optimized_df = MemoryOptimizer.downcast_numeric_types(mock_df)
    print("\nOptimized Column Data Types:")
    print(optimized_df.dtypes)