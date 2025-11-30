"""Retry mechanisms with exponential backoff and circuit breaker"""
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)
import logging
from functools import wraps
from backend.utils.logging_config import get_logger
import time

logger = get_logger(__name__)


def retry_with_backoff(
    max_attempts: int = 3,
    min_wait: int = 2,
    max_wait: int = 10,
    exceptions: tuple = (Exception,)
):
    """
    Retry decorator with exponential backoff
    
    Args:
        max_attempts: Maximum retry attempts (default: 3)
        min_wait: Minimum wait time in seconds (default: 2)
        max_wait: Maximum wait time in seconds (default: 10)
        exceptions: Tuple of exceptions to retry on
    
    Returns:
        Decorated function
    """
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(min=min_wait, max=max_wait),
        retry=retry_if_exception_type(exceptions),
        before_sleep=before_sleep_log(logger, logging.INFO),
        reraise=True
    )


class CircuitBreaker:
    """
    Circuit breaker pattern implementation
    
    Opens circuit after threshold failures, preventing further attempts
    for a timeout period.
    """
    
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        """
        Initialize circuit breaker
        
        Args:
            failure_threshold: Number of failures before opening circuit
            timeout: Timeout in seconds before attempting to close circuit
        """
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'closed'  # closed, open, half-open
        
        logger.info(f'CircuitBreaker initialized: threshold={failure_threshold}, timeout={timeout}s')
    
    def call(self, func, *args, **kwargs):
        """
        Execute function with circuit breaker protection
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
        
        Returns:
            Function result
        
        Raises:
            Exception: If circuit is open or function fails
        """
        # Check if circuit is open
        if self.state == 'open':
            # Check if timeout has passed
            if time.time() - self.last_failure_time >= self.timeout:
                logger.info('Circuit breaker: Attempting half-open state')
                self.state = 'half-open'
            else:
                raise Exception(f'Circuit breaker is OPEN. Timeout: {self.timeout}s')
        
        try:
            result = func(*args, **kwargs)
            
            # Success - reset or close circuit
            if self.state == 'half-open':
                logger.info('Circuit breaker: Closing circuit after successful call')
                self.state = 'closed'
                self.failure_count = 0
            
            return result
            
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            logger.error(f'Circuit breaker: Failure {self.failure_count}/{self.failure_threshold}')
            
            # Open circuit if threshold reached
            if self.failure_count >= self.failure_threshold:
                self.state = 'open'
                logger.error(f'Circuit breaker: OPENED after {self.failure_count} failures')
            
            raise e
    
    def reset(self):
        """Reset circuit breaker to closed state"""
        self.state = 'closed'
        self.failure_count = 0
        self.last_failure_time = None
        logger.info('Circuit breaker: Reset to closed state')
    
    def get_state(self) -> dict:
        """
        Get current circuit breaker state
        
        Returns:
            Dictionary with state information
        """
        return {
            'state': self.state,
            'failure_count': self.failure_count,
            'failure_threshold': self.failure_threshold,
            'last_failure_time': self.last_failure_time,
            'timeout': self.timeout
        }


# Global circuit breakers for different services
_circuit_breakers = {}


def get_circuit_breaker(service_name: str, failure_threshold: int = 5, timeout: int = 60) -> CircuitBreaker:
    """
    Get or create circuit breaker for a service
    
    Args:
        service_name: Name of the service
        failure_threshold: Number of failures before opening
        timeout: Timeout in seconds
    
    Returns:
        CircuitBreaker instance
    """
    if service_name not in _circuit_breakers:
        _circuit_breakers[service_name] = CircuitBreaker(failure_threshold, timeout)
    return _circuit_breakers[service_name]


def with_circuit_breaker(service_name: str, failure_threshold: int = 5, timeout: int = 60):
    """
    Decorator to add circuit breaker protection
    
    Args:
        service_name: Name of the service
        failure_threshold: Number of failures before opening
        timeout: Timeout in seconds
    
    Returns:
        Decorated function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            breaker = get_circuit_breaker(service_name, failure_threshold, timeout)
            return breaker.call(func, *args, **kwargs)
        return wrapper
    return decorator
