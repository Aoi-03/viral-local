"""
Retry mechanisms and error recovery utilities for the Viral-Local system.

This module provides robust retry logic with exponential backoff, API rate limiting
handling, and comprehensive error recovery strategies.
"""

import time
import random
import asyncio
import logging
from typing import Any, Callable, Optional, Type, Union, List, Dict
from functools import wraps
from dataclasses import dataclass
from enum import Enum

from .errors import ViralLocalError, APIError, ProcessingError


class RetryStrategy(Enum):
    """Different retry strategies available."""
    FIXED_DELAY = "fixed_delay"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    RANDOM_JITTER = "random_jitter"


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    backoff_multiplier: float = 2.0
    jitter: bool = True
    jitter_range: float = 0.1
    
    # Exception handling
    retryable_exceptions: List[Type[Exception]] = None
    non_retryable_exceptions: List[Type[Exception]] = None
    
    # Callbacks
    on_retry: Optional[Callable] = None
    on_failure: Optional[Callable] = None
    on_success: Optional[Callable] = None
    
    def __post_init__(self):
        """Set default exception lists if not provided."""
        if self.retryable_exceptions is None:
            self.retryable_exceptions = [
                ConnectionError,
                TimeoutError,
                APIError,
                ProcessingError
            ]
        
        if self.non_retryable_exceptions is None:
            self.non_retryable_exceptions = [
                ValueError,
                TypeError,
                KeyError,
                AttributeError
            ]


class RetryManager:
    """Manages retry logic and error recovery."""
    
    def __init__(self, config: Optional[RetryConfig] = None):
        """Initialize retry manager.
        
        Args:
            config: Retry configuration (uses defaults if not provided)
        """
        self.config = config or RetryConfig()
        self.logger = logging.getLogger(__name__)
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for the given attempt number.
        
        Args:
            attempt: Current attempt number (0-based)
            
        Returns:
            Delay in seconds
        """
        if self.config.strategy == RetryStrategy.FIXED_DELAY:
            delay = self.config.base_delay
        
        elif self.config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = self.config.base_delay * (self.config.backoff_multiplier ** attempt)
        
        elif self.config.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = self.config.base_delay * (attempt + 1)
        
        elif self.config.strategy == RetryStrategy.RANDOM_JITTER:
            delay = self.config.base_delay + random.uniform(0, self.config.base_delay)
        
        else:
            delay = self.config.base_delay
        
        # Apply jitter if enabled
        if self.config.jitter and self.config.strategy != RetryStrategy.RANDOM_JITTER:
            jitter = delay * self.config.jitter_range * (2 * random.random() - 1)
            delay += jitter
        
        # Ensure delay is within bounds
        delay = max(0, min(delay, self.config.max_delay))
        
        return delay
    
    def _is_retryable(self, exception: Exception) -> bool:
        """Check if an exception is retryable.
        
        Args:
            exception: Exception to check
            
        Returns:
            True if the exception should trigger a retry
        """
        # Check non-retryable exceptions first
        for exc_type in self.config.non_retryable_exceptions:
            if isinstance(exception, exc_type):
                return False
        
        # Check retryable exceptions
        for exc_type in self.config.retryable_exceptions:
            if isinstance(exception, exc_type):
                return True
        
        # Default to not retryable for unknown exceptions
        return False
    
    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute a function with retry logic.
        
        Args:
            func: Function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Result of the function execution
            
        Raises:
            Exception: The last exception if all retries fail
        """
        last_exception = None
        
        for attempt in range(self.config.max_attempts):
            try:
                result = func(*args, **kwargs)
                
                # Success callback
                if self.config.on_success:
                    self.config.on_success(attempt, result)
                
                if attempt > 0:
                    self.logger.info(f"Function succeeded after {attempt + 1} attempts")
                
                return result
                
            except Exception as e:
                last_exception = e
                
                # Check if we should retry
                if not self._is_retryable(e):
                    self.logger.error(f"Non-retryable exception: {e}")
                    raise e
                
                # Check if we have more attempts
                if attempt >= self.config.max_attempts - 1:
                    break
                
                # Calculate delay
                delay = self._calculate_delay(attempt)
                
                self.logger.warning(
                    f"Attempt {attempt + 1} failed: {e}. "
                    f"Retrying in {delay:.2f} seconds..."
                )
                
                # Retry callback
                if self.config.on_retry:
                    self.config.on_retry(attempt, e, delay)
                
                # Wait before retry
                time.sleep(delay)
        
        # All retries failed
        if self.config.on_failure:
            self.config.on_failure(self.config.max_attempts, last_exception)
        
        self.logger.error(f"All {self.config.max_attempts} attempts failed. Last error: {last_exception}")
        raise last_exception
    
    async def execute_async(self, func: Callable, *args, **kwargs) -> Any:
        """Execute an async function with retry logic.
        
        Args:
            func: Async function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Result of the function execution
            
        Raises:
            Exception: The last exception if all retries fail
        """
        last_exception = None
        
        for attempt in range(self.config.max_attempts):
            try:
                result = await func(*args, **kwargs)
                
                # Success callback
                if self.config.on_success:
                    self.config.on_success(attempt, result)
                
                if attempt > 0:
                    self.logger.info(f"Async function succeeded after {attempt + 1} attempts")
                
                return result
                
            except Exception as e:
                last_exception = e
                
                # Check if we should retry
                if not self._is_retryable(e):
                    self.logger.error(f"Non-retryable exception: {e}")
                    raise e
                
                # Check if we have more attempts
                if attempt >= self.config.max_attempts - 1:
                    break
                
                # Calculate delay
                delay = self._calculate_delay(attempt)
                
                self.logger.warning(
                    f"Async attempt {attempt + 1} failed: {e}. "
                    f"Retrying in {delay:.2f} seconds..."
                )
                
                # Retry callback
                if self.config.on_retry:
                    self.config.on_retry(attempt, e, delay)
                
                # Wait before retry
                await asyncio.sleep(delay)
        
        # All retries failed
        if self.config.on_failure:
            self.config.on_failure(self.config.max_attempts, last_exception)
        
        self.logger.error(f"All {self.config.max_attempts} async attempts failed. Last error: {last_exception}")
        raise last_exception


def retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF,
    backoff_multiplier: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: Optional[List[Type[Exception]]] = None,
    non_retryable_exceptions: Optional[List[Type[Exception]]] = None,
    on_retry: Optional[Callable] = None,
    on_failure: Optional[Callable] = None,
    on_success: Optional[Callable] = None
):
    """Decorator for adding retry logic to functions.
    
    Args:
        max_attempts: Maximum number of attempts
        base_delay: Base delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        strategy: Retry strategy to use
        backoff_multiplier: Multiplier for exponential backoff
        jitter: Whether to add random jitter to delays
        retryable_exceptions: List of exceptions that should trigger retries
        non_retryable_exceptions: List of exceptions that should not trigger retries
        on_retry: Callback function called on each retry
        on_failure: Callback function called when all retries fail
        on_success: Callback function called on successful execution
    """
    config = RetryConfig(
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=max_delay,
        strategy=strategy,
        backoff_multiplier=backoff_multiplier,
        jitter=jitter,
        retryable_exceptions=retryable_exceptions,
        non_retryable_exceptions=non_retryable_exceptions,
        on_retry=on_retry,
        on_failure=on_failure,
        on_success=on_success
    )
    
    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                retry_manager = RetryManager(config)
                return await retry_manager.execute_async(func, *args, **kwargs)
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                retry_manager = RetryManager(config)
                return retry_manager.execute(func, *args, **kwargs)
            return sync_wrapper
    
    return decorator


class RateLimiter:
    """Handles API rate limiting with request queuing."""
    
    def __init__(
        self,
        requests_per_minute: int = 60,
        requests_per_hour: Optional[int] = None,
        burst_size: Optional[int] = None
    ):
        """Initialize rate limiter.
        
        Args:
            requests_per_minute: Maximum requests per minute
            requests_per_hour: Maximum requests per hour (optional)
            burst_size: Maximum burst size (defaults to requests_per_minute)
        """
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.burst_size = burst_size or requests_per_minute
        
        self.minute_requests = []
        self.hour_requests = []
        
        self.logger = logging.getLogger(__name__)
    
    def _cleanup_old_requests(self):
        """Remove old request timestamps."""
        current_time = time.time()
        
        # Remove requests older than 1 minute
        self.minute_requests = [
            req_time for req_time in self.minute_requests
            if current_time - req_time < 60
        ]
        
        # Remove requests older than 1 hour
        if self.requests_per_hour:
            self.hour_requests = [
                req_time for req_time in self.hour_requests
                if current_time - req_time < 3600
            ]
    
    def can_make_request(self) -> bool:
        """Check if a request can be made without waiting.
        
        Returns:
            True if request can be made immediately
        """
        self._cleanup_old_requests()
        
        # Check minute limit
        if len(self.minute_requests) >= self.requests_per_minute:
            return False
        
        # Check hour limit
        if self.requests_per_hour and len(self.hour_requests) >= self.requests_per_hour:
            return False
        
        # Check burst limit
        recent_requests = [
            req_time for req_time in self.minute_requests
            if time.time() - req_time < 10  # Last 10 seconds
        ]
        if len(recent_requests) >= self.burst_size:
            return False
        
        return True
    
    def wait_time(self) -> float:
        """Calculate how long to wait before making a request.
        
        Returns:
            Wait time in seconds
        """
        self._cleanup_old_requests()
        
        if self.can_make_request():
            return 0.0
        
        wait_times = []
        
        # Check minute limit
        if len(self.minute_requests) >= self.requests_per_minute:
            oldest_request = min(self.minute_requests)
            wait_times.append(60 - (time.time() - oldest_request))
        
        # Check hour limit
        if self.requests_per_hour and len(self.hour_requests) >= self.requests_per_hour:
            oldest_request = min(self.hour_requests)
            wait_times.append(3600 - (time.time() - oldest_request))
        
        # Check burst limit
        recent_requests = [
            req_time for req_time in self.minute_requests
            if time.time() - req_time < 10
        ]
        if len(recent_requests) >= self.burst_size:
            oldest_recent = min(recent_requests)
            wait_times.append(10 - (time.time() - oldest_recent))
        
        return max(wait_times) if wait_times else 0.0
    
    def acquire(self, timeout: Optional[float] = None) -> bool:
        """Acquire permission to make a request.
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if permission acquired, False if timeout
        """
        start_time = time.time()
        
        while True:
            if self.can_make_request():
                # Record the request
                current_time = time.time()
                self.minute_requests.append(current_time)
                if self.requests_per_hour:
                    self.hour_requests.append(current_time)
                
                return True
            
            # Check timeout
            if timeout and (time.time() - start_time) >= timeout:
                return False
            
            # Wait before checking again
            wait_time = min(self.wait_time(), 1.0)  # Wait at most 1 second at a time
            time.sleep(wait_time)
    
    async def acquire_async(self, timeout: Optional[float] = None) -> bool:
        """Async version of acquire.
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if permission acquired, False if timeout
        """
        start_time = time.time()
        
        while True:
            if self.can_make_request():
                # Record the request
                current_time = time.time()
                self.minute_requests.append(current_time)
                if self.requests_per_hour:
                    self.hour_requests.append(current_time)
                
                return True
            
            # Check timeout
            if timeout and (time.time() - start_time) >= timeout:
                return False
            
            # Wait before checking again
            wait_time = min(self.wait_time(), 1.0)
            await asyncio.sleep(wait_time)


def rate_limited(
    requests_per_minute: int = 60,
    requests_per_hour: Optional[int] = None,
    timeout: Optional[float] = None
):
    """Decorator for rate limiting function calls.
    
    Args:
        requests_per_minute: Maximum requests per minute
        requests_per_hour: Maximum requests per hour
        timeout: Maximum time to wait for rate limit
    """
    rate_limiter = RateLimiter(requests_per_minute, requests_per_hour)
    
    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                if not await rate_limiter.acquire_async(timeout):
                    raise APIError(
                        "Rate limit timeout exceeded",
                        error_code="RATE_LIMIT_TIMEOUT"
                    )
                return await func(*args, **kwargs)
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                if not rate_limiter.acquire(timeout):
                    raise APIError(
                        "Rate limit timeout exceeded",
                        error_code="RATE_LIMIT_TIMEOUT"
                    )
                return func(*args, **kwargs)
            return sync_wrapper
    
    return decorator


class CircuitBreaker:
    """Circuit breaker pattern for handling cascading failures."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: Type[Exception] = Exception
    ):
        """Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Time to wait before attempting recovery
            expected_exception: Exception type that triggers the circuit breaker
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open
        
        self.logger = logging.getLogger(__name__)
    
    def can_execute(self) -> bool:
        """Check if execution is allowed.
        
        Returns:
            True if execution is allowed
        """
        if self.state == "closed":
            return True
        
        if self.state == "open":
            if self.last_failure_time and (time.time() - self.last_failure_time) > self.recovery_timeout:
                self.state = "half-open"
                self.logger.info("Circuit breaker moving to half-open state")
                return True
            return False
        
        if self.state == "half-open":
            return True
        
        return False
    
    def record_success(self):
        """Record a successful execution."""
        self.failure_count = 0
        if self.state == "half-open":
            self.state = "closed"
            self.logger.info("Circuit breaker closed after successful recovery")
    
    def record_failure(self, exception: Exception):
        """Record a failed execution.
        
        Args:
            exception: Exception that occurred
        """
        if isinstance(exception, self.expected_exception):
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "open"
                self.logger.warning(
                    f"Circuit breaker opened after {self.failure_count} failures. "
                    f"Will retry after {self.recovery_timeout} seconds."
                )
    
    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute a function with circuit breaker protection.
        
        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Result of function execution
            
        Raises:
            Exception: If circuit is open or function fails
        """
        if not self.can_execute():
            raise APIError(
                "Circuit breaker is open",
                error_code="CIRCUIT_BREAKER_OPEN",
                details={"failure_count": self.failure_count}
            )
        
        try:
            result = func(*args, **kwargs)
            self.record_success()
            return result
        except Exception as e:
            self.record_failure(e)
            raise


def circuit_breaker(
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0,
    expected_exception: Type[Exception] = Exception
):
    """Decorator for circuit breaker pattern.
    
    Args:
        failure_threshold: Number of failures before opening circuit
        recovery_timeout: Time to wait before attempting recovery
        expected_exception: Exception type that triggers the circuit breaker
    """
    breaker = CircuitBreaker(failure_threshold, recovery_timeout, expected_exception)
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return breaker.execute(func, *args, **kwargs)
        return wrapper
    
    return decorator


# Global rate limiters for common APIs
gemini_rate_limiter = RateLimiter(requests_per_minute=60, requests_per_hour=1000)
groq_rate_limiter = RateLimiter(requests_per_minute=30, requests_per_hour=500)
youtube_rate_limiter = RateLimiter(requests_per_minute=100, requests_per_hour=10000)


def get_rate_limiter(api_name: str) -> Optional[RateLimiter]:
    """Get rate limiter for a specific API.
    
    Args:
        api_name: Name of the API
        
    Returns:
        RateLimiter instance or None if not found
    """
    limiters = {
        "gemini": gemini_rate_limiter,
        "groq": groq_rate_limiter,
        "youtube": youtube_rate_limiter
    }
    
    return limiters.get(api_name.lower())