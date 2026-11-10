
import hashlib
import logging
from pathlib import Path
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("ChecksumValidator")

def calculate_sha256(file_path: Path, chunk_size: int = 8192) -> str:
    """
    Calculate the SHA-256 checksum of a file.

    Args:
        file_path (Path): Path to the file.
        chunk_size (int): Size of chunks to read the file in bytes.

    Returns:
        str: SHA-256 checksum of the file.
    """
    sha256 = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def validate_file_integrity(file_path: Path, expected_checksum: str = None) -> bool:
    """
    Validates file existence, verifies it is not a zero-byte file, 
    and optionally checks against an expected SHA-256 hash.
    
    Args:
        file_path (Path): Path to the target file.
        expected_checksum (str, optional): Expected SHA-256 hash string.
        
    Returns:
        bool: True if validation passes, False otherwise.
    """
    # 1. Existence Check
    if not file_path.exists():
        logger.error(f"File not found at path: {file_path}")
        return False

    # 2. Zero-Byte & Size Check
    file_size_bytes = file_path.stat().st_size
    logger.info(f"Inspecting file: {file_path.name} | Size: {file_size_bytes} bytes")

    if file_size_bytes == 0:
        logger.critical(f"Zero-byte check FAILED: {file_path.name} is empty (0 bytes).")
        return False

    logger.info(f"Zero-byte check PASSED for {file_path.name}")

    # 3. Optional Checksum Verification
    if expected_checksum:
        actual_checksum = calculate_sha256(file_path)
        logger.info(f"Computed SHA-256 : {actual_checksum}")
        logger.info(f"Expected SHA-256 : {expected_checksum}")
        
        if actual_checksum != expected_checksum:
            logger.critical("Checksum verification FAILED! Payload may be corrupted or tampered with.")
            return False
            
        logger.info("Checksum verification PASSED.")

    return True

def validate_payload_health(df: pd.DataFrame, source_name: str) -> bool:
    """
    Universal health validator for all non-file extraction sources 
    (Database, API, Kafka).
    """
    logger.info(f"Running payload health check for source: [{source_name}]...")

    # 1. Emptiness Check (The equivalent of a zero-byte check for DataFrames)
    if df is None or df.empty:
        logger.critical(f"[HEALTH CHECK FAILED] Source '{source_name}' returned 0 records.")
        return False

    # 2. Volume Anomaly Check (e.g., abnormally low row count)
    min_expected_rows = 1
    if len(df) < min_expected_rows:
        logger.warning(f"[WARNING] Low volume warning from '{source_name}': Only {len(df)} records received.")

    # 3. Critical Column Presence Check
    essential_cols = ["customer_id"]
    for col in essential_cols:
        if col not in df.columns:
            logger.critical(f"[HEALTH CHECK FAILED] Essential identifier column '{col}' missing from '{source_name}'.")
            return False

    logger.info(f"[HEALTH CHECK PASSED] Source '{source_name}' is healthy ({len(df):,} records loaded).")
    return True

if __name__ == "__main__":
    sample_dir = Path("data/raw")
    sample_dir.mkdir(parents=True, exist_ok=True)

    test_file = sample_dir / "test_telemetry.csv"

    if not test_file.exists():
        test_file.write_text(
            "customer_id,tenure_months,monthly_charges,total_charges,contract_type,payment_method,churn_status\n"
            "CUST-001,12,29.95,359.4,month-to-month,electronic_check,no\n",
            encoding="utf-8"
        )
    is_valid = validate_file_integrity(test_file)
    print(f"\n[RESULT] File Integrity Status: {'PASSED (Ready for Ingestion)' if is_valid else 'REJECTED'}")