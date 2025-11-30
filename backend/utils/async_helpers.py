"""Async execution helpers for agent orchestration"""
import asyncio
import time
from typing import Callable, Any
from backend.utils.logging_config import get_logger

logger = get_logger(__name__)


async def run_in_thread(func: Callable, *args, **kwargs) -> Any:
    """
    Run blocking function in thread pool
    
    Args:
        func: Blocking function to run
        *args: Function arguments
        **kwargs: Function keyword arguments
    
    Returns:
        Function result
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: func(*args, **kwargs))


async def run_parallel(*tasks):
    """
    Run multiple async tasks in parallel
    
    Args:
        *tasks: Async tasks to run
    
    Returns:
        List of results
    """
    start_time = time.time()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    elapsed = time.time() - start_time
    
    logger.info(f'Parallel execution completed: {len(tasks)} tasks in {elapsed:.2f}s')
    
    return results


def measure_async_performance(sequential_time: float, parallel_time: float) -> dict:
    """
    Measure and log async performance improvement
    
    Args:
        sequential_time: Time for sequential execution
        parallel_time: Time for parallel execution
    
    Returns:
        Performance metrics dictionary
    """
    improvement = ((sequential_time - parallel_time) / sequential_time * 100) if sequential_time > 0 else 0
    
    metrics = {
        'sequential_time': round(sequential_time, 2),
        'parallel_time': round(parallel_time, 2),
        'improvement_percent': round(improvement, 2),
        'speedup': round(sequential_time / parallel_time, 2) if parallel_time > 0 else 0
    }
    
    logger.info(f'Async performance: {metrics["improvement_percent"]}% faster', extra=metrics)
    
    return metrics
