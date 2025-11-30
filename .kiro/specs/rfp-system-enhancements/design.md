# Design Document

## Overview

This design document outlines the architecture and implementation approach for enhancing the RFP Agent System with intelligent semantic search, automated extraction, real-time web scraping, structured validation, dynamic pricing, conversation memory, and production-ready infrastructure.

The system currently uses a multi-agent architecture with four specialized agents (Main, Sales, Technical, Pricing) coordinated through LangChain. The enhancements will transform the system from a proof-of-concept with mock data and keyword matching into a production-ready platform with semantic understanding, real data integration, and enterprise-grade reliability.

**Key Design Goals:**
- Improve match accuracy from 65% to 90% through semantic search
- Reduce response time from 45s to 15s through caching and async execution
- Achieve <1% error rate through validation and retry mechanisms
- Enable production deployment with authentication, rate limiting, and monitoring

## Architecture

### High-Level Architecture

The system maintains its existing multi-agent orchestration pattern while adding new layers for intelligence, reliability, and observability:

```
┌─────────────────────────────────────────────────────────────┐
│                     API Layer (Flask)                        │
│  - Authentication (Bearer tokens)                            │
│  - Rate Limiting (per-endpoint)                              │
│  - Metrics Export (Prometheus)                               │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                  Orchestration Layer                         │
│  - Main Agent (async coordination)                           │
│  - Conversation Memory (SQLChatMessageHistory)               │
│  - Inter-Agent Validation (Pydantic schemas)                 │
│  - Error Recovery (Retry with exponential backoff)           │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
┌───────▼────────┐   ┌────────▼────────┐   ┌───────▼────────┐
│  Sales Agent   │   │ Technical Agent │   │ Pricing Agent  │
│  - Web Scraping│   │ - Semantic      │   │ - Dynamic      │
│  - Structured  │   │   Search        │   │   Margins      │
│    Parsing     │   │ - FAISS Index   │   │ - Quantity     │
│  - NER Extract │   │                 │   │   Extraction   │
└────────────────┘   └─────────────────┘   └────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                   Infrastructure Layer                       │
│  - Caching (Redis + in-memory fallback)                      │
│  - Logging (JSON structured, rotated files)                  │
│  - Database (SQLAlchemy connection pool)                     │
│  - Embeddings (sentence-transformers + FAISS)                │
└─────────────────────────────────────────────────────────────┘
```

### Component Interaction Flow

1. **Request Flow:**
   - API receives request → Authentication → Rate limit check
   - Main Agent retrieves conversation memory
   - Cache check for identical previous requests
   - Async orchestration of agent pipeline
   - Validation between each stage
   - Response compilation and caching
   - Metrics recording

2. **Agent Pipeline:**
   - Sales Agent: Web scraping → Structured parsing → NER extraction
   - Technical Agent: Semantic embedding → FAISS search → Confidence scoring
   - Pricing Agent: Quantity mapping → Dynamic margin calculation → Cost estimation

3. **Error Handling:**
   - Retry decorator on all external calls (LLM, web, database)
   - Validation schemas catch malformed outputs
   - Circuit breaker prevents cascade failures
   - Graceful degradation with partial results

## Components and Interfaces

### 1. Semantic Search Engine

**Location:** `backend/tools/sku_tools.py` - New class `SemanticMatcher`

**Responsibilities:**
- Generate embeddings for SKU catalog and RFP specifications
- Build and persist FAISS index for fast similarity search
- Calculate cosine similarity scores
- Return top-k matches above confidence threshold

**Interface:**
```python
class SemanticMatcher:
    def __init__(self, db_connection, model_name='all-MiniLM-L6-v2'):
        """Initialize with sentence-transformers model"""
        
    def build_index(self, force_rebuild=False) -> None:
        """Build FAISS index from SKU catalog"""
        
    def match(self, rfp_text: str, top_k=10, threshold=0.6) -> List[Match]:
        """
        Find semantically similar SKUs
        Returns: List of Match objects with sku_id, score, metadata
        """
        
    def get_embedding(self, text: str) -> np.ndarray:
        """Generate embedding vector for text"""
```

**Key Design Decisions:**
- Use `all-MiniLM-L6-v2` model (384 dimensions, fast, good quality)
- FAISS IndexFlatIP for exact cosine similarity
- Persist index to `data/faiss_index.bin` for fast startup
- Rebuild index on catalog changes via flag
- Normalize embeddings for cosine similarity

### 2. Quantity Extractor

**Location:** `backend/tools/rfp_tools.py` - New class `QuantityExtractor`

**Responsibilities:**
- Extract quantity mentions using spaCy NER
- Parse quantity patterns with regex
- Map quantities to SKU names using fuzzy matching
- Provide confidence scores for extractions

**Interface:**
```python
class QuantityExtractor:
    def __init__(self, spacy_model='en_core_web_sm'):
        """Initialize with spaCy model"""
        
    def extract(self, rfp_text: str, sku_names: List[str]) -> Dict[str, QuantityInfo]:
        """
        Extract quantities and map to SKUs
        Returns: {sku_name: QuantityInfo(quantity, confidence, context)}
        """
        
    def _parse_patterns(self, text: str) -> List[Tuple[int, str]]:
        """Extract (quantity, context) using regex patterns"""
        
    def _fuzzy_match_sku(self, context: str, sku_names: List[str]) -> Tuple[str, float]:
        """Match context to SKU name with confidence score"""
```

**Extraction Patterns:**
- `\d+\s*x\s*` → "10x circuit breakers"
- `quantity:\s*\d+` → "Quantity: 15"
- `\d+\s+units?` → "20 units"
- `\d+\s+(?:linear\s+)?feet` → "500 feet"

**Fuzzy Matching:**
- Use Levenshtein distance for SKU name matching
- Threshold: 80% similarity
- Fallback to quantity=1 if no match or low confidence

### 3. Web Scraper

**Location:** `backend/tools/rfp_tools.py` - Enhanced `find_rfp_online()`

**Responsibilities:**
- Search for RFPs using SerpAPI or Google Custom Search
- Scrape content from discovered URLs
- Handle multiple document formats (PDF, HTML, DOC)
- Implement rate limiting and retry logic

**Interface:**
```python
def find_rfp_online(query: str, max_results=5) -> List[RFPDocument]:
    """
    Search and scrape RFPs from web
    Returns: List of RFPDocument(url, title, content, format)
    """
    
def _construct_search_query(query: str) -> str:
    """Build search query with site filters"""
    
def _scrape_content(url: str, timeout=10) -> str:
    """Extract text content from URL"""
    
def _extract_pdf_text(pdf_bytes: bytes) -> str:
    """Extract text from PDF documents"""
```

**Search Strategy:**
- Site filters: `site:*.gov OR site:*.org OR site:*.edu`
- Keywords: "RFP", "Request for Proposal", "tender", "bid"
- Timeout: 10 seconds per request
- Retry: 3 attempts with exponential backoff
- Cache results for 1 hour

**Alternative:** Use Firecrawl API for cleaner extraction without manual parsing

### 4. Structured Parser

**Location:** `backend/tools/rfp_tools.py` - Enhanced `parse_rfp_text()`

**Responsibilities:**
- Extract structured fields from RFP text
- Validate output against Pydantic schemas
- Handle parsing errors gracefully
- Return partial results when possible

**Data Models:**
```python
from pydantic import BaseModel, Field
from typing import List, Optional

class RFPRequirement(BaseModel):
    item: str
    quantity: Optional[int] = None
    specifications: Dict[str, Any] = {}
    
class RFPParsed(BaseModel):
    title: str
    rfp_number: Optional[str] = None
    budget_min: Optional[float] = None
    budget_max: Optional[float] = None
    timeline: Optional[str] = None
    requirements: List[RFPRequirement] = []
    compliance: List[str] = []
    contact: Optional[str] = None
```

**Interface:**
```python
def parse_rfp_text(rfp_text: str) -> RFPParsed:
    """
    Parse RFP into structured format
    Uses LangChain PydanticOutputParser for schema enforcement
    """
```

**Parsing Strategy:**
- Use LLM with PydanticOutputParser for field extraction
- Semantic section detection (budget, timeline, requirements)
- Fallback to regex for common patterns
- Validate all fields before returning

### 5. Validation Layer

**Location:** `backend/utils/validation.py` - New module

**Responsibilities:**
- Define validation schemas for each agent output
- Validate data between pipeline stages
- Log validation errors with context
- Provide graceful degradation options

**Schemas:**
```python
from pydantic import BaseModel, validator

class SalesAgentOutput(BaseModel):
    status: str
    agent: str
    output: str
    parsed_data: dict
    
    @validator('status')
    def status_must_be_valid(cls, v):
        if v not in ['success', 'error']:
            raise ValueError('Invalid status')
        return v
    
    @validator('output')
    def output_not_empty(cls, v):
        if not v or len(v) < 10:
            raise ValueError('Output too short')
        return v

class TechnicalAgentOutput(BaseModel):
    status: str
    agent: str
    matches: List[dict]
    avg_match_score: float
    
    @validator('avg_match_score')
    def score_in_range(cls, v):
        if not 0 <= v <= 100:
            raise ValueError('Score out of range')
        return v

class PricingAgentOutput(BaseModel):
    status: str
    agent: str
    total_cost: float
    margin: float
    final_price: float
    breakdown: List[dict]
```

**Interface:**
```python
def validate_agent_output(output: dict, agent_type: str) -> Tuple[bool, Optional[str]]:
    """
    Validate agent output against schema
    Returns: (is_valid, error_message)
    """
    
def validate_pipeline_stage(stage_name: str, output: dict) -> None:
    """
    Validate and raise ValidationError if invalid
    Used in Main Agent orchestration
    """
```

### 6. Dynamic Pricing Engine

**Location:** `backend/tools/sku_tools.py` - New class `PricingStrategy`

**Responsibilities:**
- Calculate dynamic margins based on multiple factors
- Apply volume discounts
- Enforce margin bounds
- Log pricing decisions for transparency

**Interface:**
```python
class PricingStrategy:
    def __init__(self, base_margin=0.25, min_margin=0.15, max_margin=0.40):
        """Initialize with margin bounds"""
        
    def calculate_margin(self, 
                        total_cost: float,
                        complexity: str = 'medium',
                        competition: str = 'medium',
                        customer_type: str = 'new') -> float:
        """
        Calculate dynamic margin percentage
        Returns: margin as decimal (e.g., 0.28 for 28%)
        """
        
    def apply_volume_discount(self, total_cost: float, margin: float) -> float:
        """Apply volume-based margin adjustments"""
        
    def get_pricing_factors(self) -> dict:
        """Return dict of factors used in last calculation"""
```

**Margin Calculation Logic:**
```python
base_margin = 0.25  # 25% starting point

# Complexity adjustment
complexity_adj = {
    'simple': -0.05,
    'medium': 0.00,
    'complex': +0.10
}

# Competition adjustment
competition_adj = {
    'high': -0.08,
    'medium': 0.00,
    'low': +0.12
}

# Customer type adjustment
customer_adj = {
    'new': 0.00,
    'returning': -0.03,
    'enterprise': +0.08
}

# Volume discount
if total_cost > 100000:
    volume_adj = -0.10
elif total_cost > 50000:
    volume_adj = -0.05
else:
    volume_adj = 0.00

final_margin = base_margin + complexity_adj + competition_adj + customer_adj + volume_adj
final_margin = max(min_margin, min(final_margin, max_margin))
```

### 7. Conversation Memory

**Location:** `backend/agents/main_agent.py` - Enhanced MainAgent class

**Responsibilities:**
- Store conversation history per session
- Retrieve context for agent prompts
- Persist to database
- Manage session lifecycle

**Interface:**
```python
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import SQLChatMessageHistory

class MainAgent:
    def __init__(self, db_connection, sales_agent, technical_agent, pricing_agent):
        self.memory = self._initialize_memory()
        
    def _initialize_memory(self):
        """Create conversation memory with SQL persistence"""
        return ConversationBufferMemory(
            chat_memory=SQLChatMessageHistory(
                session_id=self.session_id,
                connection_string=self.db_connection_string
            ),
            return_messages=True
        )
        
    def orchestrate(self, rfp_input: str, session_id: str = None) -> dict:
        """Enhanced orchestration with memory"""
        # Retrieve conversation context
        context = self.memory.load_memory_variables({})
        
        # Pass context to agents
        # ... agent execution ...
        
        # Store interaction
        self.memory.save_context(
            {"input": rfp_input},
            {"output": final_response}
        )
```

### 8. Logging System

**Location:** `backend/utils/logging_config.py` - New module

**Responsibilities:**
- Configure structured JSON logging
- Rotate log files
- Log agent execution metrics
- Log LLM API calls

**Configuration:**
```python
import logging
from pythonjsonlogger import jsonlogger

def setup_logging():
    """Configure application logging"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # File handler with rotation
    file_handler = RotatingFileHandler(
        'logs/agent_activity.log',
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    
    # JSON formatter
    formatter = jsonlogger.JsonFormatter(
        '%(timestamp)s %(level)s %(name)s %(message)s'
    )
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
```

**Logging Points:**
- Agent start/end with execution time
- LLM API calls with token counts
- Cache hits/misses
- Validation failures
- Errors with stack traces

### 9. Caching Layer

**Location:** `backend/utils/cache.py` - New module

**Responsibilities:**
- Cache agent results
- Generate cache keys
- Manage TTL
- Track metrics

**Interface:**
```python
import redis
import hashlib
from functools import wraps

class CacheManager:
    def __init__(self, redis_url=None):
        """Initialize with Redis or in-memory fallback"""
        
    def get(self, key: str) -> Optional[Any]:
        """Retrieve cached value"""
        
    def set(self, key: str, value: Any, ttl=3600) -> None:
        """Store value with TTL in seconds"""
        
    def generate_key(self, agent_name: str, input_text: str) -> str:
        """Generate SHA256 cache key"""
        return hashlib.sha256(f"{agent_name}:{input_text}".encode()).hexdigest()
        
    def get_metrics(self) -> dict:
        """Return cache hit/miss statistics"""

def cached(ttl=3600):
    """Decorator for caching agent methods"""
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            cache_key = cache_manager.generate_key(
                self.__class__.__name__,
                str(args) + str(kwargs)
            )
            
            # Check cache
            cached_result = cache_manager.get(cache_key)
            if cached_result:
                return cached_result
            
            # Execute and cache
            result = func(self, *args, **kwargs)
            cache_manager.set(cache_key, result, ttl)
            return result
        return wrapper
    return decorator
```

**Usage:**
```python
class TechnicalAgent:
    @cached(ttl=3600)
    def analyze(self, rfp_text: str) -> dict:
        # ... analysis logic ...
```

### 10. Retry Mechanism

**Location:** `backend/utils/retry.py` - New module

**Responsibilities:**
- Retry failed operations
- Exponential backoff
- Circuit breaker pattern
- Logging retry attempts

**Interface:**
```python
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=2, max=10),
    retry=retry_if_exception_type((RequestException, TimeoutError)),
    before_sleep=log_retry_attempt
)
def retry_wrapper(func):
    """Decorator for retrying operations"""
    return func

class CircuitBreaker:
    def __init__(self, failure_threshold=5, timeout=60):
        """Initialize circuit breaker"""
        
    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection"""
```

**Usage:**
```python
@retry_wrapper
def find_rfp_online(query: str) -> str:
    # ... web scraping logic ...
```

## Data Models

### Database Schema Extensions

**New Tables:**

```sql
-- Conversation history
CREATE TABLE conversation_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    message_type TEXT NOT NULL,  -- 'human' or 'ai'
    content TEXT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_session (session_id)
);

-- Cache entries (if not using Redis)
CREATE TABLE cache_entries (
    cache_key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    expires_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- API tokens
CREATE TABLE api_tokens (
    token_hash TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    role TEXT DEFAULT 'user',  -- 'user' or 'admin'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP NOT NULL
);

-- Metrics (if not using Prometheus)
CREATE TABLE metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    metric_name TEXT NOT NULL,
    metric_value REAL NOT NULL,
    labels TEXT,  -- JSON
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Pydantic Models

All data models are defined in `backend/models/` directory:

- `rfp_models.py`: RFPParsed, RFPRequirement, RFPDocument
- `agent_models.py`: SalesAgentOutput, TechnicalAgentOutput, PricingAgentOutput
- `match_models.py`: Match, QuantityInfo
- `pricing_models.py`: PricingFactors, CostBreakdown

## Correc
tness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system-essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Semantic Search Properties

**Property 1: Embedding dimensionality consistency**
*For any* RFP specification text, generating an embedding SHALL produce a vector of exactly 384 dimensions (for all-MiniLM-L6-v2 model).
**Validates: Requirements 1.1**

**Property 2: Cosine similarity bounds**
*For any* pair of embedding vectors, the calculated cosine similarity SHALL return a value between -1 and 1 inclusive.
**Validates: Requirements 1.2**

**Property 3: Similarity threshold filtering**
*For any* search result set, all returned matches SHALL have similarity scores greater than or equal to 0.6.
**Validates: Requirements 1.3**

### Quantity Extraction Properties

**Property 4: Quantity extraction completeness**
*For any* RFP text containing quantity patterns ("10x", "Quantity: 10", "10 units"), the extraction system SHALL identify and extract the numeric quantity value.
**Validates: Requirements 2.1, 2.2**

**Property 5: Fuzzy matching produces closest SKU**
*For any* extracted quantity with context, the fuzzy matching system SHALL map it to the SKU name with highest similarity score above 80% threshold.
**Validates: Requirements 2.3**

**Property 6: Default quantity fallback**
*For any* SKU without an extracted quantity, the system SHALL assign a default quantity value of exactly 1.
**Validates: Requirements 2.4**

**Property 7: Confidence score bounds**
*For any* extracted quantity, the returned confidence score SHALL be a value between 0 and 1 inclusive.
**Validates: Requirements 2.5**

### Web Scraping Properties

**Property 8: Search query includes site filters**
*For any* search query constructed by the system, the query string SHALL contain site filters for government and organization domains (*.gov, *.org, *.edu).
**Validates: Requirements 3.2**

**Property 9: Retry with exponential backoff**
*For any* failed web request that triggers retry logic, the delay between attempts SHALL increase exponentially with minimum 2 seconds and maximum 10 seconds, and SHALL not exceed 3 total attempts.
**Validates: Requirements 3.4**

**Property 10: Request timeout enforcement**
*For any* web request, the system SHALL terminate the request if it exceeds 10 seconds duration.
**Validates: Requirements 3.6**

**Property 11: Multi-format content extraction**
*For any* document in PDF, HTML, or DOC format, the scraping system SHALL successfully extract text content without format-specific errors.
**Validates: Requirements 3.5**

### Structured Parsing Properties

**Property 12: Required field extraction**
*For any* RFP text, the parsing system SHALL extract all required fields (title, budget_min, budget_max, timeline, requirements, compliance) and return them in the structured output.
**Validates: Requirements 4.1**

**Property 13: Schema validation enforcement**
*For any* parsing output, the system SHALL validate against the Pydantic schema before returning, and SHALL raise ValidationError if schema constraints are violated.
**Validates: Requirements 4.2**

**Property 14: Partial results on validation failure**
*For any* parsing operation where validation fails, the system SHALL return successfully extracted fields with error indicators for failed fields, rather than returning nothing.
**Validates: Requirements 4.3**

**Property 15: Pydantic model return type**
*For any* parsing operation, the return value SHALL be an instance of RFPParsed Pydantic model, not a dictionary.
**Validates: Requirements 4.4**

### Validation Properties

**Property 16: Agent output schema validation**
*For any* agent execution that completes, the system SHALL validate the output against the agent-specific Pydantic schema.
**Validates: Requirements 5.1**

**Property 17: Required field presence check**
*For any* agent output validation, the system SHALL verify presence of status field, non-empty output, and all required fields defined in the schema.
**Validates: Requirements 5.2**

**Property 18: Pipeline halt on critical validation failure**
*For any* validation failure on critical fields (status, required outputs), the pipeline execution SHALL stop immediately and SHALL not proceed to subsequent stages.
**Validates: Requirements 5.3**

**Property 19: Validation error logging**
*For any* validation failure, the system SHALL log an error message containing the failed stage name, field name, and validation error details.
**Validates: Requirements 5.4**

**Property 20: Structured error response**
*For any* validation failure, the system SHALL return a structured error response containing status='error', failed_stage, and error_message fields.
**Validates: Requirements 5.5**

### Dynamic Pricing Properties

**Property 21: Complexity margin adjustment bounds**
*For any* margin calculation with complexity parameter, the complexity adjustment SHALL be within the range [-5%, +10%] where simple=-5%, medium=0%, complex=+10%.
**Validates: Requirements 6.1**

**Property 22: Competition margin adjustment bounds**
*For any* margin calculation with competition parameter, the competition adjustment SHALL be within the range [-8%, +12%] where high=-8%, medium=0%, low=+12%.
**Validates: Requirements 6.2**

**Property 23: Customer type margin adjustment bounds**
*For any* margin calculation with customer_type parameter, the customer adjustment SHALL be within the range [-3%, +8%] where new=0%, returning=-3%, enterprise=+8%.
**Validates: Requirements 6.3**

**Property 24: Volume discount application**
*For any* order with total_cost exceeding $50,000, the system SHALL apply a volume discount adjustment, where $50k-$100k receives -5% and >$100k receives -10%.
**Validates: Requirements 6.4**

**Property 25: Margin bounds enforcement**
*For any* final margin calculation, the result SHALL be clamped to minimum 15% and maximum 40%, regardless of input adjustments.
**Validates: Requirements 6.5**

**Property 26: Pricing factors logging**
*For any* margin calculation, the system SHALL log all adjustment factors (complexity, competition, customer_type, volume) used in the calculation.
**Validates: Requirements 6.6**

### Conversation Memory Properties

**Property 27: Session ID uniqueness**
*For any* two distinct RFP sessions created by the system, their session identifiers SHALL be unique.
**Validates: Requirements 7.1**

**Property 28: Input-output pair storage**
*For any* agent execution, the system SHALL store the input-output pair in conversation memory associated with the current session.
**Validates: Requirements 7.2**

**Property 29: Conversation history retrieval**
*For any* agent execution with existing conversation history, the system SHALL retrieve and provide the history to the agent before processing.
**Validates: Requirements 7.3**

**Property 30: Memory persistence to database**
*For any* conversation memory operation (save or load), the system SHALL persist the data to database using SQLChatMessageHistory.
**Validates: Requirements 7.4**

**Property 31: Session memory isolation**
*For any* new session, the conversation memory SHALL be empty and SHALL not contain messages from previous sessions.
**Validates: Requirements 7.5**

### Logging Properties

**Property 32: Agent execution logging completeness**
*For any* agent execution, the system SHALL log all required fields: timestamp, input_length, output_status, and execution_time.
**Validates: Requirements 8.1**

**Property 33: Error stack trace logging**
*For any* error that occurs, the system SHALL log the error message along with the complete stack trace.
**Validates: Requirements 8.2**

**Property 34: LLM API call logging**
*For any* LLM API call, the system SHALL log the model name, token count, and estimated cost.
**Validates: Requirements 8.3**

**Property 35: JSON log format**
*For any* log event, the system SHALL write the log entry in JSON format to both file and console handlers.
**Validates: Requirements 8.5**

### Caching Properties

**Property 36: Cache key generation consistency**
*For any* agent input, the cache key SHALL be generated as SHA256 hash of the concatenation of agent_name and input_text, producing a deterministic 64-character hexadecimal string.
**Validates: Requirements 9.1**

**Property 37: Cache hit returns stored value**
*For any* cache lookup where the key exists and TTL has not expired, the system SHALL return the cached value without re-executing the operation.
**Validates: Requirements 9.2**

**Property 38: Cache TTL setting**
*For any* value stored in cache, the TTL SHALL be set to exactly 3600 seconds (1 hour).
**Validates: Requirements 9.3**

**Property 39: Cache metrics tracking**
*For any* cache operation (hit or miss), the system SHALL increment the corresponding metric counter.
**Validates: Requirements 9.4**

**Property 40: Cache invalidation mechanism**
*For any* cache invalidation request, the system SHALL provide methods to clear either a specific cache key or all cached entries.
**Validates: Requirements 9.5**

### Retry Mechanism Properties

**Property 41: Maximum retry attempts**
*For any* operation that encounters retryable exceptions, the system SHALL attempt the operation at most 3 times total (1 initial + 2 retries).
**Validates: Requirements 10.1**

**Property 42: Exponential backoff timing**
*For any* retry sequence, the delay before each retry SHALL follow exponential backoff with minimum 2 seconds, maximum 10 seconds, and SHALL increase exponentially between attempts.
**Validates: Requirements 10.2**

**Property 43: Retry attempt logging**
*For any* retry attempt, the system SHALL log the attempt number and the failure reason that triggered the retry.
**Validates: Requirements 10.3**

**Property 44: Fallback execution after max retries**
*For any* operation that exceeds maximum retry attempts and has a defined fallback mechanism, the system SHALL execute the fallback.
**Validates: Requirements 10.4**

**Property 45: Circuit breaker activation**
*For any* operation that fails repeatedly (5+ consecutive failures), the circuit breaker SHALL open and SHALL reject subsequent attempts for the timeout period.
**Validates: Requirements 10.5**

### Async Execution Properties

**Property 46: Async LLM call wrapping**
*For any* blocking LLM API call, the system SHALL wrap the call in asyncio.to_thread to prevent blocking the event loop.
**Validates: Requirements 11.2**

**Property 47: Performance timing logging**
*For any* async pipeline execution, the system SHALL log timing comparison showing both parallel and sequential execution times.
**Validates: Requirements 11.4**

### Database Connection Pooling Properties

**Property 48: Connection health check**
*For any* connection acquired from the pool, the system SHALL perform a health check (pool_pre_ping) before returning the connection.
**Validates: Requirements 12.3**

**Property 49: Connection return to pool**
*For any* connection that is released, the system SHALL return it to the connection pool for reuse rather than closing it.
**Validates: Requirements 12.4**

**Property 50: Pool metrics exposure**
*For any* request to pool metrics, the system SHALL return accurate counts of active and idle connections.
**Validates: Requirements 12.5**

### Rate Limiting Properties

**Property 51: Global rate limit enforcement**
*For any* IP address, the system SHALL enforce a maximum of 100 requests per hour, returning HTTP 429 for requests exceeding this limit.
**Validates: Requirements 13.1**

**Property 52: Analyze endpoint rate limit**
*For any* IP address making requests to /api/rfp/analyze, the system SHALL enforce a maximum of 10 requests per minute.
**Validates: Requirements 13.2**

**Property 53: Catalog endpoint rate limit**
*For any* IP address making requests to /api/catalog, the system SHALL enforce a maximum of 50 requests per minute.
**Validates: Requirements 13.3**

**Property 54: Rate limit response format**
*For any* request that exceeds rate limits, the system SHALL return HTTP 429 status with a retry-after header indicating when the client can retry.
**Validates: Requirements 13.4**

**Property 55: Admin rate limit bypass**
*For any* request with valid admin authentication, the system SHALL not apply rate limiting.
**Validates: Requirements 13.5**

### Metrics Properties

**Property 56: Request count recording**
*For any* API request, the system SHALL increment the request count metric for the corresponding endpoint.
**Validates: Requirements 14.2**

**Property 57: Response time histogram recording**
*For any* API response, the system SHALL record the response time in the histogram metric for the corresponding endpoint.
**Validates: Requirements 14.3**

**Property 58: Agent duration recording**
*For any* agent execution, the system SHALL record the execution duration in the metric for that agent type.
**Validates: Requirements 14.4**

**Property 59: Error rate recording**
*For any* agent error, the system SHALL increment the error count metric for that agent type.
**Validates: Requirements 14.5**

**Property 60: Cache ratio recording**
*For any* cache operation, the system SHALL update the hit/miss ratio metrics.
**Validates: Requirements 14.6**

**Property 61: Connection pool usage recording**
*For any* database connection operation, the system SHALL update the connection pool usage metrics.
**Validates: Requirements 14.7**

### Authentication Properties

**Property 62: Bearer token verification**
*For any* request to a protected endpoint, the system SHALL extract and verify the Bearer token from the Authorization header.
**Validates: Requirements 15.1**

**Property 63: Invalid token response**
*For any* request with an invalid or missing Bearer token, the system SHALL return HTTP 401 status.
**Validates: Requirements 15.3**

**Property 64: Insufficient permissions response**
*For any* authenticated request where the user lacks required permissions, the system SHALL return HTTP 403 status.
**Validates: Requirements 15.4**

**Property 65: Token expiration setting**
*For any* newly issued token, the system SHALL set the expiration time to exactly 24 hours from issuance.
**Validates: Requirements 15.5**

**Property 66: Authentication attempt logging**
*For any* authentication attempt (successful or failed), the system SHALL log the attempt with timestamp, user identifier, and outcome.
**Validates: Requirements 15.6**

## Error Handling

### Error Categories

1. **Validation Errors**
   - Schema validation failures
   - Missing required fields
   - Type mismatches
   - **Handling:** Return structured error response, log details, stop pipeline

2. **External Service Errors**
   - LLM API failures
   - Web scraping timeouts
   - Search API errors
   - **Handling:** Retry with exponential backoff, fallback to cached/default values, circuit breaker

3. **Data Errors**
   - Database connection failures
   - Cache unavailability
   - File I/O errors
   - **Handling:** Connection pool retry, in-memory cache fallback, graceful degradation

4. **Authentication/Authorization Errors**
   - Invalid tokens
   - Expired tokens
   - Insufficient permissions
   - **Handling:** Return appropriate HTTP status (401/403), log attempt, no retry

5. **Rate Limiting Errors**
   - Quota exceeded
   - **Handling:** Return 429 with retry-after header, no retry

### Error Response Format

All errors return consistent structure:

```json
{
  "status": "error",
  "error_type": "ValidationError",
  "message": "Agent output validation failed",
  "details": {
    "failed_stage": "technical",
    "field": "matches",
    "reason": "Field required"
  },
  "timestamp": "2025-01-15T10:30:00Z",
  "request_id": "req_abc123"
}
```

### Graceful Degradation Strategy

- **Semantic Search Unavailable:** Fall back to keyword matching with warning
- **Quantity Extraction Fails:** Use default quantity=1 for all SKUs
- **Web Scraping Fails:** Use provided RFP text directly
- **Cache Unavailable:** Continue without caching, log warning
- **Memory Unavailable:** Operate in stateless mode

## Testing Strategy

### Unit Testing

**Framework:** pytest

**Coverage Areas:**
- Individual function correctness (embedding generation, cache key hashing, margin calculation)
- Edge cases (empty inputs, null values, boundary conditions)
- Error conditions (invalid schemas, missing fields, malformed data)
- Mock external dependencies (LLM API, web requests, database)

**Example Unit Tests:**
```python
def test_semantic_matcher_embedding_dimensions():
    """Verify embeddings have correct dimensionality"""
    matcher = SemanticMatcher()
    embedding = matcher.get_embedding("test text")
    assert embedding.shape == (384,)

def test_quantity_extractor_default_fallback():
    """Verify default quantity when extraction fails"""
    extractor = QuantityExtractor()
    result = extractor.extract("No quantities here", ["SKU-001"])
    assert result["SKU-001"].quantity == 1

def test_pricing_strategy_margin_bounds():
    """Verify margin stays within bounds"""
    strategy = PricingStrategy()
    # Test extreme inputs
    margin = strategy.calculate_margin(
        total_cost=1000,
        complexity='complex',
        competition='low',
        customer_type='enterprise'
    )
    assert 0.15 <= margin <= 0.40
```

### Property-Based Testing

**Framework:** Hypothesis (Python)

**Configuration:**
- Minimum 100 iterations per property test
- Shrinking enabled for minimal failing examples
- Stateful testing for pipeline workflows

**Property Test Examples:**

```python
from hypothesis import given, strategies as st

@given(st.text(min_size=1, max_size=1000))
def test_property_embedding_dimensionality(rfp_text):
    """Property 1: Embedding dimensionality consistency"""
    # Feature: rfp-system-enhancements, Property 1: Embedding dimensionality consistency
    matcher = SemanticMatcher()
    embedding = matcher.get_embedding(rfp_text)
    assert embedding.shape == (384,), f"Expected 384 dimensions, got {embedding.shape}"

@given(st.floats(min_value=-1, max_value=1), st.floats(min_value=-1, max_value=1))
def test_property_cosine_similarity_bounds(vec1_val, vec2_val):
    """Property 2: Cosine similarity bounds"""
    # Feature: rfp-system-enhancements, Property 2: Cosine similarity bounds
    vec1 = np.array([vec1_val] * 384)
    vec2 = np.array([vec2_val] * 384)
    similarity = cosine_similarity(vec1, vec2)
    assert -1 <= similarity <= 1

@given(st.lists(st.floats(min_value=0, max_value=1), min_size=1, max_size=50))
def test_property_similarity_threshold_filtering(scores):
    """Property 3: Similarity threshold filtering"""
    # Feature: rfp-system-enhancements, Property 3: Similarity threshold filtering
    matches = [{"score": s} for s in scores]
    filtered = filter_matches_by_threshold(matches, threshold=0.6)
    assert all(m["score"] >= 0.6 for m in filtered)

@given(st.text(min_size=1), st.lists(st.text(min_size=1), min_size=1))
def test_property_cache_key_consistency(agent_name, inputs):
    """Property 36: Cache key generation consistency"""
    # Feature: rfp-system-enhancements, Property 36: Cache key generation consistency
    cache = CacheManager()
    for input_text in inputs:
        key1 = cache.generate_key(agent_name, input_text)
        key2 = cache.generate_key(agent_name, input_text)
        assert key1 == key2  # Deterministic
        assert len(key1) == 64  # SHA256 hex length
        assert all(c in '0123456789abcdef' for c in key1)  # Valid hex

@given(st.floats(min_value=0, max_value=1000000))
def test_property_margin_bounds_enforcement(total_cost):
    """Property 25: Margin bounds enforcement"""
    # Feature: rfp-system-enhancements, Property 25: Margin bounds enforcement
    strategy = PricingStrategy()
    # Test all extreme combinations
    for complexity in ['simple', 'medium', 'complex']:
        for competition in ['high', 'medium', 'low']:
            for customer in ['new', 'returning', 'enterprise']:
                margin = strategy.calculate_margin(
                    total_cost, complexity, competition, customer
                )
                assert 0.15 <= margin <= 0.40

@given(st.integers(min_value=1, max_value=10))
def test_property_retry_max_attempts(num_failures):
    """Property 41: Maximum retry attempts"""
    # Feature: rfp-system-enhancements, Property 41: Maximum retry attempts
    attempt_count = 0
    
    @retry_wrapper
    def failing_operation():
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count <= num_failures:
            raise RequestException("Simulated failure")
        return "success"
    
    try:
        failing_operation()
    except:
        pass
    
    assert attempt_count <= 3  # Max 3 attempts
```

### Integration Testing

**Scope:** End-to-end pipeline workflows

**Test Scenarios:**
1. Complete RFP processing pipeline (Sales → Technical → Pricing → Main)
2. Cache hit/miss scenarios
3. Retry and fallback mechanisms
4. Authentication and rate limiting
5. Database connection pooling under load

**Example Integration Test:**
```python
def test_complete_rfp_pipeline():
    """Test full pipeline with all agents"""
    db = get_db()
    main_agent = MainAgent(db, sales_agent, technical_agent, pricing_agent)
    
    rfp_input = "Need 10 circuit breakers for manufacturing facility"
    result = main_agent.orchestrate(rfp_input)
    
    assert result['status'] == 'success'
    assert 'sales' in result['stages']
    assert 'technical' in result['stages']
    assert 'pricing' in result['stages']
    assert result['final_response']['pricing_proposal']['final_price'] > 0
```

### Performance Testing

**Tools:** pytest-benchmark, locust

**Metrics:**
- Response time per endpoint
- Throughput (requests/second)
- Cache hit rate
- Database connection pool utilization

**Targets:**
- Average response time: <15 seconds
- P95 response time: <25 seconds
- Cache hit rate: >60%
- Error rate: <1%

### Load Testing

**Tool:** Locust

**Scenarios:**
- 100 concurrent users
- Sustained load for 10 minutes
- Spike testing (sudden 10x increase)

**Success Criteria:**
- No crashes or memory leaks
- Rate limiting functions correctly
- Connection pool handles load
- Graceful degradation under extreme load

## Deployment Considerations

### Environment Variables

```bash
# LLM Configuration
GROQ_API_KEY=your_api_key_here

# Search APIs
SERPAPI_KEY=your_serpapi_key
FIRECRAWL_API_KEY=your_firecrawl_key

# Database
DATABASE_URL=sqlite:///data/rfp_system.db

# Redis Cache
REDIS_URL=redis://localhost:6379/0

# Authentication
API_TOKEN_SECRET=your_secret_key_here

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/agent_activity.log

# Rate Limiting
RATE_LIMIT_ENABLED=true
RATE_LIMIT_STORAGE_URL=redis://localhost:6379/1
```

### Infrastructure Requirements

- **Python:** 3.9+
- **Redis:** 6.0+ (for caching and rate limiting)
- **Disk Space:** 2GB (for FAISS index, logs, database)
- **Memory:** 4GB minimum (8GB recommended for production)
- **CPU:** 2 cores minimum (4 cores recommended)

### Monitoring and Alerting

**Prometheus Metrics:**
- Request rate and latency
- Error rates by agent
- Cache hit/miss ratios
- Database connection pool usage
- LLM API call costs

**Alerts:**
- Error rate >5% for 5 minutes
- Response time P95 >30 seconds
- Cache hit rate <40%
- Database connection pool exhaustion
- Rate limit violations >100/hour

### Security Considerations

1. **API Tokens:** Store hashed in environment, rotate every 90 days
2. **Rate Limiting:** Prevent abuse and DDoS
3. **Input Validation:** Sanitize all user inputs
4. **HTTPS Only:** Enforce TLS for all API endpoints
5. **Secrets Management:** Use environment variables, never commit secrets
6. **Audit Logging:** Log all authentication attempts and admin actions

## Migration Strategy

### Phase 1: Core Intelligence (Week 1)
- Implement semantic search with FAISS
- Add quantity extraction with spaCy
- Add validation layer
- **Risk:** Model download size, embedding generation time
- **Mitigation:** Pre-build index, cache embeddings

### Phase 2: Data Integration (Week 2)
- Implement web scraping
- Add structured parsing
- Enhance logging
- **Risk:** External API rate limits, scraping failures
- **Mitigation:** Implement retry logic, fallback to mock data

### Phase 3: Reliability (Week 3)
- Add dynamic pricing
- Implement conversation memory
- Add caching layer
- Add retry mechanisms
- **Risk:** Redis dependency, memory overhead
- **Mitigation:** In-memory fallback, connection pooling

### Phase 4: Production Readiness (Week 4)
- Implement async execution
- Add connection pooling
- Add rate limiting
- Add metrics and monitoring
- Add authentication
- **Risk:** Breaking changes to API, performance regression
- **Mitigation:** Versioned API endpoints, load testing

### Rollback Plan

Each phase is independently deployable. If issues arise:
1. Revert to previous Docker image
2. Disable feature flags for new functionality
3. Fall back to mock data/keyword matching
4. Monitor error rates and performance metrics

## Dependencies

### Python Packages

```
# Core Framework
flask>=3.0.0
langchain>=0.1.0
langchain-openai>=0.0.5
langchain-community>=0.0.20

# Semantic Search
sentence-transformers>=2.2.0
faiss-cpu>=1.7.4
numpy>=1.24.0

# NLP & Text Processing
spacy>=3.7.0
fuzzywuzzy>=0.18.0
python-levenshtein>=0.23.0

# Web Scraping
google-search-results>=2.4.0
beautifulsoup4>=4.12.0
requests>=2.31.0
firecrawl-py>=0.0.5

# Data Validation
pydantic>=2.0.0

# Caching & Memory
redis>=5.0.0

# Database
sqlalchemy>=2.0.0

# Error Handling
tenacity>=8.2.0

# Production
flask-limiter>=3.5.0
prometheus-flask-exporter>=0.22.0
flask-httpauth>=4.8.0
python-json-logger>=2.0.0

# Async
aiohttp>=3.9.0

# Testing
pytest>=7.4.0
pytest-asyncio>=0.21.0
pytest-benchmark>=4.0.0
hypothesis>=6.92.0
locust>=2.20.0
```

### External Services

- **Groq API:** LLM inference
- **SerpAPI or Google Custom Search:** RFP discovery
- **Firecrawl (optional):** Clean web scraping
- **Redis:** Caching and rate limiting
- **Prometheus:** Metrics collection

### Model Downloads

```bash
# spaCy English model
python -m spacy download en_core_web_sm

# sentence-transformers model (auto-downloaded on first use)
# all-MiniLM-L6-v2 (~90MB)
```
