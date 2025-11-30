"""Pricing Agent: Cost Estimation and Proposal Pricing"""
import os
import json
import time
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from backend.tools.sku_tools import estimate_cost
from backend.tools.rfp_tools import extract_quantities_from_rfp
from backend.utils.logging_config import get_logger, log_agent_execution
from backend.utils.retry import retry_with_backoff
from backend.utils.cache import cached
from openai import RateLimitError, APIError

logger = get_logger(__name__)


class PricingAgent:
    """
    Pricing Agent responsible for:
    - Calculating total costs
    - Estimating quantities
    - Applying appropriate margins
    - Generating pricing breakdowns
    """
    
    def __init__(self, db_connection, use_quantity_extraction=True):
        self.db_connection = db_connection
        self.use_quantity_extraction = use_quantity_extraction
        self.llm = self._initialize_llm()
        logger.info(f'Pricing Agent initialized with quantity_extraction={use_quantity_extraction}')
    
    def _initialize_llm(self):
        """Initialize Groq LLM"""
        api_key = os.getenv('GROQ_API_KEY')
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        
        return ChatOpenAI(
            model="openai/gpt-oss-20b",
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
            temperature=0.1
        )
    
    @retry_with_backoff(
        max_attempts=3,
        min_wait=2,
        max_wait=10,
        exceptions=(RateLimitError, APIError)
    )
    def _call_llm(self, matched_skus: list, cost_data: dict):
        """Call LLM with retry logic"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Pricing Agent specialized in cost estimation and proposal pricing for industrial equipment.

Your responsibilities:
1. Calculate total costs based on matched SKUs
2. Estimate realistic quantities for RFP requirements
3. Apply appropriate profit margins (typically 20-30%)
4. Generate detailed pricing breakdowns
5. Consider volume discounts and project complexity

When calculating pricing, consider:
- Unit costs from catalog
- Estimated quantities based on project scope
- Standard industry margins
- Competitive positioning
- Project complexity factors"""),
            ("human", """Review and provide insights on this pricing proposal:

Matched SKUs:
{matched_skus}

Cost Breakdown:
{cost_breakdown}

Provide pricing recommendations and any competitive positioning notes.""")
        ])
        
        chain = prompt | self.llm
        return chain.invoke({
            "matched_skus": json.dumps(matched_skus, indent=2),
            "cost_breakdown": json.dumps(cost_data, indent=2)
        })
    
    @cached(ttl=3600)  # Cache for 1 hour
    def analyze(self, matched_skus: list, rfp_text: str = None) -> dict:
        """Calculate pricing for matched SKUs"""
        start_time = time.time()
        
        try:
            # Extract quantities from RFP if enabled and text provided
            if self.use_quantity_extraction and rfp_text:
                logger.info('Extracting quantities from RFP text')
                quantities = extract_quantities_from_rfp(rfp_text, matched_skus)
                logger.info(f'Extracted quantities for {len(quantities)} SKUs')
            else:
                logger.info('Using default quantity estimation')
                quantities = None
            
            # Get cost estimation
            cost_data = estimate_cost(matched_skus, self.db_connection, quantities=quantities)
            
            # Use LLM to provide pricing insights (with retry)
            result = self._call_llm(matched_skus, cost_data)
            
            execution_time = time.time() - start_time
            
            # Log execution
            log_agent_execution(
                logger,
                agent_name='pricing',
                input_length=len(json.dumps(matched_skus)),
                output_status='success',
                execution_time=execution_time,
                total_cost=cost_data['total_cost'],
                final_price=cost_data['final_price'],
                quantity_extraction_used=self.use_quantity_extraction and rfp_text is not None
            )
            
            return {
                'status': 'success',
                'agent': 'pricing',
                'output': result.content if hasattr(result, 'content') else str(result),
                'total_cost': cost_data['total_cost'],
                'margin': cost_data['margin'],
                'final_price': cost_data['final_price'],
                'breakdown': cost_data['breakdown']
            }
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f'Pricing Agent error: {e}', exc_info=True)
            log_agent_execution(
                logger,
                agent_name='pricing',
                input_length=len(json.dumps(matched_skus)) if matched_skus else 0,
                output_status='error',
                execution_time=execution_time,
                error=str(e)
            )
            return {
                'status': 'error',
                'agent': 'pricing',
                'error': str(e)
            }