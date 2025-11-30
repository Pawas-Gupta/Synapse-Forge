"""Sales Agent: RFP Discovery and Summarization"""
import os
import time
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from backend.tools.rfp_tools import find_rfp_online, parse_rfp_text
from backend.utils.logging_config import get_logger, log_agent_execution
from backend.utils.retry import retry_with_backoff, with_circuit_breaker
from backend.utils.cache import cached
from openai import RateLimitError, APIError

logger = get_logger(__name__)


class SalesAgent:
    """
    Sales Agent responsible for:
    - RFP discovery and retrieval
    - High-level requirement extraction
    - Project summarization
    """
    
    def __init__(self, db_connection, use_web_scraping=True, use_structured_parsing=True):
        self.db_connection = db_connection
        self.use_web_scraping = use_web_scraping
        self.use_structured_parsing = use_structured_parsing
        self.llm = self._initialize_llm()
        logger.info(f'Sales Agent initialized: web_scraping={use_web_scraping}, structured_parsing={use_structured_parsing}')
    
    def _initialize_llm(self):
        """Initialize Groq LLM"""
        api_key = os.getenv('GROQ_API_KEY')
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        
        return ChatOpenAI(
            model="openai/gpt-oss-20b",
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
            temperature=0.3
        )
    
    @retry_with_backoff(
        max_attempts=3,
        min_wait=2,
        max_wait=10,
        exceptions=(RateLimitError, APIError)
    )
    def _call_llm(self, rfp_text: str):
        """Call LLM with retry logic"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Sales Agent specialized in RFP discovery and summarization for B2B manufacturers.

Your responsibilities:
1. Identify and retrieve relevant RFPs
2. Extract high-level requirements (technical specs, timeline, budget)
3. Summarize project scope and key deliverables
4. Identify potential risks or special requirements

When analyzing an RFP, focus on:
- Project overview and objectives
- Technical requirements summary
- Timeline and deadlines
- Budget constraints
- Compliance requirements"""),
            ("human", "Analyze and summarize this RFP:\n\n{rfp_text}")
        ])
        
        chain = prompt | self.llm
        return chain.invoke({"rfp_text": rfp_text})
    
    @cached(ttl=3600)  # Cache for 1 hour
    def analyze(self, rfp_input: str) -> dict:
        """Analyze RFP and return summary"""
        start_time = time.time()
        
        try:
            # First, try to find RFP online if it looks like a query
            if self.use_web_scraping and len(rfp_input) < 200:
                logger.info('Using web scraping for RFP discovery')
                rfp_text = find_rfp_online(rfp_input)
            else:
                rfp_text = rfp_input
            
            # Parse the RFP
            if self.use_structured_parsing:
                logger.info('Using structured parsing')
                parsed_data = parse_rfp_text(rfp_text, use_llm=False)
            else:
                parsed_data = {}
            
            # Use LLM to summarize (with retry)
            result = self._call_llm(rfp_text)
            
            execution_time = time.time() - start_time
            
            # Log execution
            log_agent_execution(
                logger,
                agent_name='sales',
                input_length=len(rfp_input),
                output_status='success',
                execution_time=execution_time,
                web_scraping_used=self.use_web_scraping and len(rfp_input) < 200,
                structured_parsing_used=self.use_structured_parsing
            )
            
            return {
                'status': 'success',
                'agent': 'sales',
                'output': result.content if hasattr(result, 'content') else str(result),
                'parsed_data': parsed_data
            }
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f'Sales Agent error: {e}', exc_info=True)
            log_agent_execution(
                logger,
                agent_name='sales',
                input_length=len(rfp_input),
                output_status='error',
                execution_time=execution_time,
                error=str(e)
            )
            return {
                'status': 'error',
                'agent': 'sales',
                'error': str(e)
            }
