# Stage 2: Data Ingestion & Low-Level Hygiene

## Overview
Welcome to **Stage 2** of the Machine Learning Engineering Lifecycle. This module establishes a production-grade ingestion layer capable of handling multi-source extractions, rigorous file integrity checks, schema contract enforcement with automated quarantine, and memory optimization via type downcasting.


## Learning Objectives
By completing this stage, intermediate engineers will learn how to:
1. **Validate File Integrity:** Implement zero-byte checks and SHA-256 cryptographic checksums before touching payloads.
2. **Handle Multi-Source Extractions:** Ingest data across flat files (CSV, JSON, Parquet), databases, rate-limited REST APIs, and Kafka streams.
3. **Enforce Data Contracts:** Intercept schema drift and route corrupted or out-of-bounds records into a Dead-Letter Queue (Quarantine).
4. **Optimize RAM Footprint:** Aggressively downcast numerical data types and leverage out-of-core processing engines to prevent memory overflows.

```text
stage-02-data-ingestion/
├── README.md
├── config/
│   └── O01_schema_contract.yaml        # Schema rules and bounds contract
├── data/
│   ├── raw/                           # Landing zone (Bronze layer)
│   └── quarantine/                    # Dead-letter queue for invalid records
└── src/
    ├── hygiene/
    │   ├── O02_checksum_validator.py  # File existence, size & hash validation
    │   └── O07_schema_enforcer.py     # Contract validation & quarantine routing
    ├── extractors/
    │   ├── O03_file_batch_extractor.py# CSV, JSON, Parquet batch parsing
    │   ├── O04_db_extractor.py        # SQL/NoSQL connection pooling & chunking
    │   ├── O05_api_scraper.py         # REST APIs with backoff & rate-limiting
    │   └── O06_kafka_consumer.py      # Real-time streaming consumer groups
    ├── optimization/
    │   └── O08_memory_optimizer.py    # Type downcasting & out-of-core processing
    └── O09_orchestrator.py            # Main end-to-end pipeline execution runner
```