"""Caching layer with Redis and in-memory fallback"""
import hashlib
import json
import time
from typing import Any, Optional
from functools import wraps
from backend.utils.logging_config import get_logger

logger = get_logger(__name__)


class CacheManager:
    """
    Cache manager with Redis and in-memory fallback
    
    Generates cache keys using SHA256 hash and supports TTL.
    """
    
    def __init__(self, redis_url: Optional[str] = None):
        """
        Initialize cache manager
        
        Args:
            redis_url: Redis connection URL (optional)
        """
        self.redis_client = None
        self.memory_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Try to connect to Redis
        if redis_url:
            try:
                import redis
                self.redis_client = redis.from_url(redis_url)
                self.redis_client.ping()
                logger.info(f'Connected to Redis at {redis_url}')
            except ImportError:
                logger.warning('redis package not installed, using in-memory cache')
            except Exception as e:
                logger.warning(f'Failed to connect to Redis: {e}, using in-memory cache')
        else:
            logger.info('No Redis URL provided, using in-memory cache')
    
    def generate_key(self, agent_name: str, input_text: str) -> str:
        """
        Generate SHA256 cache key
        
        Args:
            agent_name: Name of the agent
            input_text: Input text
        
        Returns:
            64-character hexadecimal cache key
        """
        key_string = f"{agent_name}:{input_text}"
        return hashlib.sha256(key_string.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve cached value
        
        Args:
            key: Cache key
        
        Returns:
            Cached value or None
        """
        # Try Redis first
        if self.redis_client:
            try:
                value = self.redis_client.get(key)
                if value:
                    self.cache_hits += 1
                    logger.debug(f'Cache hit (Redis): {key[:16]}...')
                    return json.loads(value)
                else:
                    self.cache_misses += 1
                    return None
            except Exception as e:
                logger.error(f'Redis get error: {e}')
        
        # Fallback to memory cache
        if key in self.memory_cache:
            entry = self.memory_cache[key]
            # Check TTL
            if entry['expires_at'] > time.time():
                self.cache_hits += 1
                logger.debug(f'Cache hit (memory): {key[:16]}...')
                return entry['value']
            else:
                # Expired
                del self.memory_cache[key]
        
        self.cache_misses += 1
        return None
    
    def set(self, key: str, value: Any, ttl: int = 3600) -> None:
        """
        Store value with TTL
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (default: 3600 = 1 hour)
        """
        # Try Redis first
        if self.redis_client:
            try:
                self.redis_client.setex(key, ttl, json.dumps(value))
                logger.debug(f'Cached to Redis: {key[:16]}... (TTL: {ttl}s)')
                return
            except Exception as e:
                logger.error(f'Redis set error: {e}')
        
        # Fallback to memory cache
        self.memory_cache[key] = {
            'value': value,
            'expires_at': time.time() + ttl
        }
        logger.debug(f'Cached to memory: {key[:16]}... (TTL: {ttl}s)')
    
    def invalidate(self, key: str) -> None:
        """
        Invalidate specific cache entry
        
        Args:
            key: Cache key to invalidate
        """
        if self.redis_client:
            try:
                self.redis_client.delete(key)
            except Exception as e:
                logger.error(f'Redis delete error: {e}')
        
        if key in self.memory_cache:
            del self.memory_cache[key]
        
        logger.info(f'Invalidated cache: {key[:16]}...')
    
    def clear_all(self) -> None:
        """Clear all cached entries"""
        if self.redis_client:
            try:
                self.redis_client.flushdb()
                logger.info('Cleared Redis cache')
            except Exception as e:
                logger.error(f'Redis flush error: {e}')
        
        self.memory_cache.clear()
        logger.info('Cleared memory cache')
    
    def get_metrics(self) -> dict:
        """
        Return cache hit/miss statistics
        
        Returns:
            Dictionary with cache metrics
        """
        total = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total * 100) if total > 0 else 0
        
        return {
            'hits': self.cache_hits,
            'misses': self.cache_misses,
            'total_requests': total,
            'hit_rate': round(hit_rate, 2),
            'backend': 'redis' if self.redis_client else 'memory',
            'memory_entries': len(self.memory_cache)
        }


# Global cache manager instance
_cache_manager = None


def get_cache_manager(redis_url: Optional[str] = None) -> CacheManager:
    """
    Get or create global cache manager
    
    Args:
        redis_url: Redis connection URL (optional)
    
    Returns:
        CacheManager instance
    """
    global _cache_manager
    if _cache_manager is None:
        import os
        redis_url = redis_url or os.getenv('REDIS_URL')
        _cache_manager = CacheManager(redis_url)
    return _cache_manager


def cached(ttl: int = 3600):
    """
    Decorator for caching agent methods
    
    Args:
        ttl: Time-to-live in seconds (default: 3600 = 1 hour)
    
    Returns:
        Decorated function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            cache_manager = get_cache_manager()
            
            # Generate cache key
            agent_name = self.__class__.__name__
            input_str = str(args) + str(kwargs)
            cache_key = cache_manager.generate_key(agent_name, input_str)
            
            # Check cache
            cached_result = cache_manager.get(cache_key)
            if cached_result is not None:
                logger.info(f'{agent_name}: Cache hit')
                return cached_result
            
            # Execute function
            logger.info(f'{agent_name}: Cache miss, executing')
            result = func(self, *args, **kwargs)
            
            # Cache result
            cache_manager.set(cache_key, result, ttl)
            
            return result
        return wrapper
    return decorator
