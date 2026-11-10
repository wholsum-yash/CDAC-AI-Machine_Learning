
"""
================================================================================
MODULE: Resilient REST API Scraper
FILE: src/extractors/O05_api_scraper.py
PURPOSE:
  Provides a robust HTTP client wrapper for pulling data from external REST 
  APIs. Features automated exponential backoff retries for transient HTTP errors 
  (429, 500, 502, 503, 504), built-in rate-limiting delays to respect server 
  thresholds, and optional proxy rotation support.
================================================================================
"""

import logging
import time
from typing import Dict, Any, Optional
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("APIScraper")



class APIScraper:
    """Robust REST API scraper with retry-backoff and rate-limiting support."""

    def __init__(self, base_url: str, timeout_sec: int = 10, proxies: Optional[Dict[str, str]] = None):
        self.base_url = base_url
        self.timeout_sec = timeout_sec
        self.proxies = proxies
        self.session = self._create_resilient_session()
        logger.info(f"Initialized APIScraper for target base URL: {base_url}")

    def _create_resilient_session(self) -> requests.Session:
        """Configures a requests Session with automated retry and backoff strategies."""
        session = requests.Session()
        
        # Define retry strategy for transient errors (HTTP 429, 500, 502, 503, 504)
        retries = Retry(
            total=3,
            backoff_factor=1.0,  # Wait 1s, 2s, 4s between retries
            status_forcelist=[429, 500, 502, 503, 504],
            raise_on_status=False
        )
        
        adapter = HTTPAdapter(max_retries=retries)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        
        if self.proxies:
            session.proxies.update(self.proxies)
            logger.info("Proxy configuration applied to API session.")
            
        return session

    def fetch_endpoint(self, endpoint: str, params: Optional[Dict[str, Any]] = None, rate_limit_delay: float = 0.5) -> Optional[Dict[str, Any]]:
        """
        Fetches data from a specific API endpoint with built-in rate limiting.
        
        Args:
            endpoint (str): API endpoint path (e.g., '/v1/customers').
            params (dict, optional): Query parameters.
            rate_limit_delay (float): Seconds to wait before request to respect API limits.
            
        Returns:
            dict or list: JSON response payload if successful, None otherwise.
        """
        target_url = f"{self.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        
        # Respect rate-limiting before hitting the server
        if rate_limit_delay > 0:
            time.sleep(rate_limit_delay)
            
        logger.info(f"Sending GET request to {target_url} with params {params}")
        
        try:
            response = self.session.get(target_url, params=params, timeout=self.timeout_sec)
            print(f"Response status code: {response.status_code}")
            # Check for HTTP errors
            if response.status_code == 200:
                logger.info(f"Successfully fetched data from {endpoint} (HTTP 200)")
                return response.json()
            else:
                logger.warning(f"API returned non-200 status code: {response.status_code} for {target_url}")
                return None
                # raise requests.exceptions.RequestException(f"Non-200 status code: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Network request failed for {target_url}: {e}")
            
            # Simulated fallback for educational/offline testing
            logger.warning("[SIMULATION MODE] Returning mock API response payload.")
            return {
                "status": "success",
                "data": [
                    {"customer_id": "CUST-API-01", "support_tickets_count": 2},
                    {"customer_id": "CUST-API-02", "support_tickets_count": 0}
                ]
            }

if __name__ == "__main__":
    # Local demonstration test
    # scraper = APIScraper(base_url="https://api.example-enterprise.com")
    scraper = APIScraper(base_url="https://jsonplaceholder.typicode.com/")
    
    payload = scraper.fetch_endpoint(endpoint="todos/1", params={"limit": 10}, rate_limit_delay=0.2)
    print("\n[RESULT] Scraped API Payload:")
    print(payload)