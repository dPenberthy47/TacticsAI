"""
TacticsAI Base API Client
Production-quality HTTP client with rate limiting, retry logic, and caching.
"""

import hashlib
import json
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, TypeVar, Generic
from dataclasses import dataclass
from enum import Enum

import requests
from requests.exceptions import HTTPError, Timeout, ConnectionError, RequestException
from loguru import logger

from .config import API_URLS, APIConfig


# =============================================================================
# Type Definitions
# =============================================================================

T = TypeVar("T")


class CachePolicy(Enum):
    """Cache behavior options."""
    USE_CACHE = "use_cache"  # Use cached response if available
    REFRESH = "refresh"  # Bypass cache, fetch fresh data
    CACHE_ONLY = "cache_only"  # Only return cached data, don't fetch


@dataclass
class APIResponse(Generic[T]):
    """Wrapper for API responses with metadata."""
    data: T
    status_code: int
    from_cache: bool
    url: str
    elapsed_ms: float


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 30.0
    exponential_base: float = 2.0


# =============================================================================
# Exceptions
# =============================================================================

class APIClientError(Exception):
    """Base exception for API client errors."""
    pass


class RateLimitError(APIClientError):
    """Raised when rate limit is exceeded."""
    def __init__(self, retry_after: Optional[float] = None):
        self.retry_after = retry_after
        super().__init__(f"Rate limit exceeded. Retry after: {retry_after}s")


class NotFoundError(APIClientError):
    """Raised when resource is not found (404)."""
    pass


class TimeoutError(APIClientError):
    """Raised when request times out."""
    pass


class MaxRetriesExceededError(APIClientError):
    """Raised when max retry attempts are exhausted."""
    pass


# =============================================================================
# Base API Client
# =============================================================================

class BaseAPIClient(ABC):
    """
    Base API client with rate limiting, retry logic, and response caching.
    
    Features:
    - Rate limiting with configurable delays between requests
    - Retry logic with exponential backoff (default 3 attempts)
    - Response caching to disk using content-based hashing
    - Comprehensive error handling for common HTTP errors
    - Structured logging with loguru
    
    Usage:
        class MyClient(BaseAPIClient):
            def _get_api_config(self) -> APIConfig:
                return API_URLS["myapi"]
            
            def fetch_data(self, endpoint: str) -> dict:
                return self.get(endpoint).data
    """
    
    # Class-level cache directory (relative to backend/data/)
    CACHE_DIR = Path(__file__).parent / "cache"
    
    # Default timeout for requests (seconds)
    DEFAULT_TIMEOUT = 30
    
    def __init__(
        self,
        retry_config: Optional[RetryConfig] = None,
        timeout: int = DEFAULT_TIMEOUT,
        cache_enabled: bool = True,
    ):
        """
        Initialize the API client.
        
        Args:
            retry_config: Configuration for retry behavior
            timeout: Request timeout in seconds
            cache_enabled: Whether to enable response caching
        """
        self._retry_config = retry_config or RetryConfig()
        self._timeout = timeout
        self._cache_enabled = cache_enabled
        self._last_request_time: float = 0.0
        self._session = requests.Session()
        
        # Set up default headers
        self._session.headers.update({
            "User-Agent": "TacticsAI/1.0 (Football Analytics Engine)",
            "Accept": "application/json, text/html, */*",
            "Accept-Language": "en-US,en;q=0.9",
        })
        
        # Ensure cache directory exists
        if self._cache_enabled:
            self.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        
        logger.debug(
            f"Initialized {self.__class__.__name__} | "
            f"timeout={timeout}s | cache={cache_enabled}"
        )
    
    @abstractmethod
    def _get_api_config(self) -> APIConfig:
        """Return the API configuration for this client."""
        pass
    
    @property
    def base_url(self) -> str:
        """Get the base URL for this API."""
        return self._get_api_config().base_url
    
    @property
    def rate_limit_seconds(self) -> float:
        """Get the rate limit delay for this API."""
        return self._get_api_config().rate_limit_seconds
    
    # -------------------------------------------------------------------------
    # Rate Limiting
    # -------------------------------------------------------------------------
    
    def _enforce_rate_limit(self) -> None:
        """
        Enforce rate limiting by sleeping if necessary.
        
        Ensures minimum time between requests based on API configuration.
        """
        if self._last_request_time == 0:
            return
        
        elapsed = time.time() - self._last_request_time
        wait_time = self.rate_limit_seconds - elapsed
        
        if wait_time > 0:
            logger.debug(f"Rate limiting: waiting {wait_time:.2f}s")
            time.sleep(wait_time)
    
    def _record_request_time(self) -> None:
        """Record the timestamp of the current request."""
        self._last_request_time = time.time()
    
    # -------------------------------------------------------------------------
    # Caching
    # -------------------------------------------------------------------------
    
    def _get_cache_key(self, url: str, params: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate a unique cache key for a request.
        
        Args:
            url: The request URL
            params: Optional query parameters
            
        Returns:
            SHA256 hash of the request signature
        """
        cache_data = {"url": url, "params": params or {}}
        cache_str = json.dumps(cache_data, sort_keys=True)
        return hashlib.sha256(cache_str.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get the file path for a cache entry."""
        return self.CACHE_DIR / f"{cache_key}.json"
    
    def _read_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """
        Read cached response from disk.
        
        Args:
            cache_key: The cache key to look up
            
        Returns:
            Cached data if found, None otherwise
        """
        cache_path = self._get_cache_path(cache_key)
        
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            logger.debug(f"Cache hit: {cache_key[:12]}...")
            return cached
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Cache read error: {e}")
            return None
    
    def _write_cache(self, cache_key: str, data: Any, metadata: Dict[str, Any]) -> None:
        """
        Write response to cache.
        
        Args:
            cache_key: The cache key
            data: Response data to cache
            metadata: Additional metadata (url, status_code, etc.)
        """
        cache_path = self._get_cache_path(cache_key)
        
        try:
            cache_entry = {
                "data": data,
                "metadata": metadata,
                "cached_at": time.time(),
            }
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cache_entry, f, indent=2, default=str)
            logger.debug(f"Cached response: {cache_key[:12]}...")
        except (IOError, TypeError) as e:
            logger.warning(f"Cache write error: {e}")
    
    def clear_cache(self) -> int:
        """
        Clear all cached responses.
        
        Returns:
            Number of cache entries deleted
        """
        count = 0
        for cache_file in self.CACHE_DIR.glob("*.json"):
            try:
                cache_file.unlink()
                count += 1
            except IOError as e:
                logger.warning(f"Failed to delete cache file: {e}")
        
        logger.info(f"Cleared {count} cache entries")
        return count
    
    # -------------------------------------------------------------------------
    # Error Handling
    # -------------------------------------------------------------------------
    
    def _handle_response_errors(self, response: requests.Response) -> None:
        """
        Check response for errors and raise appropriate exceptions.
        
        Args:
            response: The HTTP response to check
            
        Raises:
            NotFoundError: If resource not found (404)
            RateLimitError: If rate limited (429)
            HTTPError: For other HTTP errors
        """
        if response.status_code == 404:
            raise NotFoundError(f"Resource not found: {response.url}")
        
        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            retry_seconds = float(retry_after) if retry_after else None
            raise RateLimitError(retry_after=retry_seconds)
        
        # Raise for other 4xx/5xx errors
        response.raise_for_status()
    
    def _calculate_backoff_delay(self, attempt: int) -> float:
        """
        Calculate exponential backoff delay for retry.
        
        Args:
            attempt: Current attempt number (0-indexed)
            
        Returns:
            Delay in seconds before next retry
        """
        delay = self._retry_config.base_delay_seconds * (
            self._retry_config.exponential_base ** attempt
        )
        return min(delay, self._retry_config.max_delay_seconds)
    
    # -------------------------------------------------------------------------
    # HTTP Methods
    # -------------------------------------------------------------------------
    
    def get(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        cache_policy: CachePolicy = CachePolicy.USE_CACHE,
    ) -> APIResponse[Any]:
        """
        Perform a GET request with rate limiting, caching, and retry logic.
        
        Args:
            endpoint: API endpoint (will be appended to base_url)
            params: Optional query parameters
            headers: Optional additional headers
            cache_policy: How to handle caching for this request
            
        Returns:
            APIResponse containing the response data and metadata
            
        Raises:
            MaxRetriesExceededError: If all retry attempts fail
            NotFoundError: If resource not found
            TimeoutError: If request times out
            APIClientError: For other API errors
        """
        # Build full URL
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        # Check cache first (if enabled and policy allows)
        cache_key = self._get_cache_key(url, params)
        
        if self._cache_enabled and cache_policy in (CachePolicy.USE_CACHE, CachePolicy.CACHE_ONLY):
            cached = self._read_cache(cache_key)
            if cached:
                return APIResponse(
                    data=cached["data"],
                    status_code=cached["metadata"].get("status_code", 200),
                    from_cache=True,
                    url=url,
                    elapsed_ms=0,
                )
            elif cache_policy == CachePolicy.CACHE_ONLY:
                raise APIClientError(f"No cached data available for: {url}")
        
        # Perform request with retries
        last_error: Optional[Exception] = None
        
        for attempt in range(self._retry_config.max_attempts):
            try:
                # Enforce rate limiting
                self._enforce_rate_limit()
                
                logger.debug(
                    f"Request: GET {url} | attempt {attempt + 1}/{self._retry_config.max_attempts}"
                )
                
                start_time = time.time()
                response = self._session.get(
                    url,
                    params=params,
                    headers=headers,
                    timeout=self._timeout,
                )
                elapsed_ms = (time.time() - start_time) * 1000
                
                self._record_request_time()
                
                # Check for errors
                self._handle_response_errors(response)
                
                # Parse response
                try:
                    data = response.json()
                except json.JSONDecodeError:
                    data = response.text
                
                # Cache successful response
                if self._cache_enabled:
                    self._write_cache(
                        cache_key,
                        data,
                        {"url": url, "status_code": response.status_code},
                    )
                
                logger.info(f"GET {url} | {response.status_code} | {elapsed_ms:.0f}ms")
                
                return APIResponse(
                    data=data,
                    status_code=response.status_code,
                    from_cache=False,
                    url=url,
                    elapsed_ms=elapsed_ms,
                )
            
            except NotFoundError:
                # Don't retry 404s
                raise
            
            except RateLimitError as e:
                last_error = e
                wait_time = e.retry_after or self._calculate_backoff_delay(attempt)
                logger.warning(f"Rate limited, waiting {wait_time:.1f}s before retry")
                time.sleep(wait_time)
            
            except Timeout as e:
                last_error = TimeoutError(f"Request timed out: {url}")
                logger.warning(f"Timeout on attempt {attempt + 1}: {e}")
                if attempt < self._retry_config.max_attempts - 1:
                    delay = self._calculate_backoff_delay(attempt)
                    time.sleep(delay)
            
            except ConnectionError as e:
                last_error = APIClientError(f"Connection error: {e}")
                logger.warning(f"Connection error on attempt {attempt + 1}: {e}")
                if attempt < self._retry_config.max_attempts - 1:
                    delay = self._calculate_backoff_delay(attempt)
                    time.sleep(delay)
            
            except HTTPError as e:
                last_error = APIClientError(f"HTTP error: {e}")
                logger.error(f"HTTP error: {e}")
                if attempt < self._retry_config.max_attempts - 1:
                    delay = self._calculate_backoff_delay(attempt)
                    time.sleep(delay)
            
            except RequestException as e:
                last_error = APIClientError(f"Request failed: {e}")
                logger.error(f"Request error: {e}")
                if attempt < self._retry_config.max_attempts - 1:
                    delay = self._calculate_backoff_delay(attempt)
                    time.sleep(delay)
        
        # All retries exhausted
        raise MaxRetriesExceededError(
            f"Failed after {self._retry_config.max_attempts} attempts: {last_error}"
        )
    
    def get_raw(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> requests.Response:
        """
        Perform a raw GET request without caching (for binary/streaming content).
        
        Args:
            endpoint: API endpoint
            params: Optional query parameters
            headers: Optional additional headers
            
        Returns:
            Raw requests.Response object
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        self._enforce_rate_limit()
        
        response = self._session.get(
            url,
            params=params,
            headers=headers,
            timeout=self._timeout,
        )
        
        self._record_request_time()
        self._handle_response_errors(response)
        
        return response
    
    def close(self) -> None:
        """Close the underlying session."""
        self._session.close()
        logger.debug(f"Closed {self.__class__.__name__} session")
    
    def __enter__(self) -> "BaseAPIClient":
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()


# =============================================================================
# Concrete Client Implementations
# =============================================================================

class FBrefClient(BaseAPIClient):
    """Client for FBref football statistics."""
    
    def _get_api_config(self) -> APIConfig:
        return API_URLS["fbref"]


class StatsBombClient(BaseAPIClient):
    """Client for StatsBomb open data (via GitHub)."""
    
    def _get_api_config(self) -> APIConfig:
        return API_URLS["statsbomb"]


class UnderstatClient(BaseAPIClient):
    """Client for Understat expected goals data."""
    
    def _get_api_config(self) -> APIConfig:
        return API_URLS["understat"]


class TransferMarktClient(BaseAPIClient):
    """Client for TransferMarkt player/team data."""
    
    def _get_api_config(self) -> APIConfig:
        return API_URLS["transfermarkt"]

