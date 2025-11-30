"""Technical Agent: SKU Matching and Specification Analysis"""
import os
import json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from backend.tools.sku_tools import match_sku, match_sku_semantic, get_catalog_summary
from backend.utils.logging_config import get_logger, log_agent_execution
from backend.utils.retry import retry_with_backoff
from backend.utils.cache import cached
from openai import RateLimitError, APIError
import time

logger = get_logger(__name__)


class TechnicalAgent:
    """
    Technical Agent responsible for:
    - Analyzing technical specifications
    - Matching requirements to SKU catalog
    - Calculating specification match scores
    """
    
    def __init__(self, db_connection, use_semantic_search=True):
        self.db_connection = db_connection
        self.use_semantic_search = use_semantic_search
        self.llm = self._initialize_llm()
        logger.info(f'Technical Agent initialized with semantic_search={use_semantic_search}')
    
    def _initialize_llm(self):
        """Initialize Groq LLM"""
        api_key = os.getenv('GROQ_API_KEY')
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        
        return ChatOpenAI(
            model="openai/gpt-oss-20b",
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
            temperature=0.2
        )
    
    @retry_with_backoff(
        max_attempts=3,
        min_wait=2,
        max_wait=10,
        exceptions=(RateLimitError, APIError)
    )
    def _call_llm(self, rfp_text: str, match_data: dict, catalog_summary: dict):
        """Call LLM with retry logic"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Technical Agent specialized in product matching and specification analysis for industrial equipment.

Your responsibilities:
1. Analyze technical requirements from RFPs
2. Match requirements against SKU catalog database
3. Calculate specification match confidence scores
4. Identify best-fit products for each requirement
5. Flag any requirements that cannot be met

When matching products, consider:
- Voltage and current ratings
- Physical specifications
- Compliance requirements (UL, NEC)
- Performance characteristics
- Compatibility with existing systems"""),
            ("human", """Analyze this RFP and provide technical insights:

RFP Text:
{rfp_text}

Matched Products:
{matches}

Catalog Summary:
{summary}

Provide a technical analysis of the matches and any recommendations.""")
        ])
        
        chain = prompt | self.llm
        return chain.invoke({
            "rfp_text": rfp_text,
            "matches": json.dumps(match_data['matches'], indent=2),
            "summary": json.dumps(catalog_summary, indent=2)
        })
    
    @cached(ttl=3600)  # Cache for 1 hour
    def analyze(self, rfp_text: str) -> dict:
        """Analyze RFP and match products"""
        start_time = time.time()
        
        try:
            # Get raw match data using semantic or keyword matching
            if self.use_semantic_search:
                logger.info('Using semantic search for SKU matching')
                match_data = match_sku_semantic(rfp_text, self.db_connection, top_k=10, threshold=0.6)
            else:
                logger.info('Using keyword search for SKU matching')
                match_data = match_sku(rfp_text, self.db_connection)
            
            catalog_summary = get_catalog_summary(self.db_connection)
            
            # Use LLM to analyze and provide insights (with retry)
            result = self._call_llm(rfp_text, match_data, catalog_summary)
            
            execution_time = time.time() - start_time
            
            # Log execution
            log_agent_execution(
                logger,
                agent_name='technical',
                input_length=len(rfp_text),
                output_status='success',
                execution_time=execution_time,
                matches_found=match_data['total_items_matched'],
                avg_match_score=match_data['avg_match_score'],
                search_method='semantic' if self.use_semantic_search else 'keyword'
            )
            
            return {
                'status': 'success',
                'agent': 'technical',
                'output': result.content if hasattr(result, 'content') else str(result),
                'matches': match_data['matches'],
                'avg_match_score': match_data['avg_match_score'],
                'total_matched': match_data['total_items_matched']
            }
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f'Technical Agent error: {e}', exc_info=True)
            log_agent_execution(
                logger,
                agent_name='technical',
                input_length=len(rfp_text),
                output_status='error',
                execution_time=execution_time,
                error=str(e)
            )
            return {
                'status': 'error',
                'agent': 'technical',
                'error': str(e)
            }