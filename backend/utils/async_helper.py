"""Async execution helpers for agent operations"""
import asyncio
import time
from typing import Callable, Any, List
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


async def run_parallel(*tasks) -> List[Any]:
    """
    Run multiple async tasks in parallel
    
    Args:
        *tasks: Async tasks to run
    
    Returns:
        List of results
    """
    start_time = time.time()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    execution_time = time.time() - start_time
    
    logger.info(f'Parallel execution completed: {len(tasks)} tasks in {execution_time:.2f}s')
    
    return results


def run_async(coro):
    """
    Run async coroutine in sync context
    
    Args:
        coro: Coroutine to run
    
    Returns:
        Coroutine result
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(coro)


class AsyncAgentWrapper:
    """
    Wrapper to make synchronous agents async-compatible
    
    Allows gradual migration to async without breaking existing code.
    """
    
    def __init__(self, agent):
        """
        Initialize async wrapper
        
        Args:
            agent: Synchronous agent instance
        """
        self.agent = agent
        self.agent_name = agent.__class__.__name__
    
    async def analyze_async(self, *args, **kwargs):
        """
        Run agent analyze method asynchronously
        
        Args:
            *args: Agent arguments
            **kwargs: Agent keyword arguments
        
        Returns:
            Agent result
        """
        logger.info(f'{self.agent_name}: Running async')
        return await run_in_thread(self.agent.analyze, *args, **kwargs)
    
    def analyze(self, *args, **kwargs):
        """
        Synchronous analyze method (for compatibility)
        
        Args:
            *args: Agent arguments
            **kwargs: Agent keyword arguments
        
        Returns:
            Agent result
        """
        return self.agent.analyze(*args, **kwargs)


async def orchestrate_async(main_agent, rfp_input: str, callback=None):
    """
    Async orchestration of agent pipeline
    
    Args:
        main_agent: Main agent instance
        rfp_input: RFP input text
        callback: Optional callback function
    
    Returns:
        Workflow result
    """
    start_time = time.time()
    
    # Stage 1: Sales Agent (sequential - needs output for next stage)
    sales_result = await run_in_thread(main_agent.sales_agent.analyze, rfp_input)
    
    if sales_result['status'] != 'success':
        return {'status': 'error', 'error': 'Sales agent failed'}
    
    rfp_text = sales_result.get('output', rfp_input)
    
    # Stage 2 & 3: Technical and Pricing can potentially run in parallel
    # But pricing needs technical results, so we run sequentially for now
    technical_result = await run_in_thread(main_agent.technical_agent.analyze, rfp_text)
    
    if technical_result['status'] != 'success':
        return {'status': 'error', 'error': 'Technical agent failed'}
    
    matched_skus = technical_result.get('matches', [])
    pricing_result = await run_in_thread(
        main_agent.pricing_agent.analyze,
        matched_skus,
        rfp_text=rfp_text
    )
    
    total_time = time.time() - start_time
    
    logger.info(f'Async orchestration completed in {total_time:.2f}s')
    
    return {
        'status': 'success',
        'stages': {
            'sales': sales_result,
            'technical': technical_result,
            'pricing': pricing_result
        },
        'execution_time': total_time,
        'async_mode': True
    }
