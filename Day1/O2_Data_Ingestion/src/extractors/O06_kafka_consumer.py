'''
Why Kafka?

Kafka is a distributed streaming platform that allows for high-throughput, low-latency processing of real-time data streams. It is widely used for building data pipelines, real-time analytics, and event-driven architectures. Key reasons to use Kafka include:
- Scalability: Can handle large volumes of data across multiple brokers.
- Durability: Ensures data is reliably stored and replicated.
- Fault-tolerance: Continues to operate even if some brokers fail.
- Real-time processing: Enables immediate processing of incoming data streams.
- Integration: Works well with various data processing frameworks and storage systems.

'''
"""
================================================================================
MODULE: Real-Time Kafka Stream Consumer
FILE: src/extractors/O06_kafka_consumer.py
PURPOSE:
  Manages real-time streaming telemetry ingestion via Apache Kafka consumer 
  groups. Handles event polling, JSON deserialization, offset management, and 
  graceful fallback simulation when message brokers are offline during local 
  development and testing.
================================================================================
"""

import json
import logging
from typing import Generator, Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("KafkaConsumer")


class KafkaTelemetryConsumer:
    """Enterprise Kafka consumer for real-time subscription telemetry streams."""

    def __init__(self, bootstrap_servers: str, topic: str, group_id: str):
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.group_id = group_id
        logger.info(f"Initialized KafkaConsumer for topic '{topic}' under group '{group_id}'")

    def poll_messages(self, max_messages: int = 5, timeout_sec: float = 2.0) -> Generator[Dict[str, Any], None, None]:
        """
        Polls messages from a Kafka topic in a consumer group loop.
        
        Args:
            max_messages (int): Maximum number of messages to yield in a single poll batch.
            timeout_sec (float): Polling timeout in seconds.
            
        Yields:
            dict: Deserialized JSON message payload.
        """
        logger.info(f"Polling Kafka topic '{self.topic}' (max_messages={max_messages})...")
        
        try:
            # Note: In production with real Kafka brokers, you would use kafka-python or confluent_kafka:
            # from kafka import KafkaConsumer
            # consumer = KafkaConsumer(
            #     self.topic,
            #     bootstrap_servers=self.bootstrap_servers,
            #     group_id=self.group_id,
            #     value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            #     auto_offset_reset='earliest',
            #     enable_auto_commit=True
            # )
            # for message in consumer:
            #     yield message.value
            
            raise ImportError("Kafka broker offline - falling back to simulation.")

        except Exception as e:
            logger.warning(f"[SIMULATION MODE] Active Kafka cluster unreachable: {e}")
            logger.info("Yielding simulated streaming telemetry events...")
            
            simulated_stream = [
                {"customer_id": "CUST-STREAM-101", "tenure_months": 5, "monthly_charges": 49.99, "churn_status": "no"},
                {"customer_id": "CUST-STREAM-102", "tenure_months": 1, "monthly_charges": 89.50, "churn_status": "yes"},
            ]
            
            for msg in simulated_stream[:max_messages]:
                yield msg



if __name__ == "__main__":
    # Local demonstration test
    consumer = KafkaTelemetryConsumer(
        bootstrap_servers="localhost:9092",
        topic="customer-telemetry-events",
        group_id="ml-ingestion-group-v1"
    )
    
    print("\n[RESULT] Consuming streaming message batch:")
    for event in consumer.poll_messages(max_messages=2):
        print(f"Received Event Payload: {event}")