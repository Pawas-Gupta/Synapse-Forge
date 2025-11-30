# Implementation Plan

- [x] 1. Set up infrastructure and utilities
  - Create directory structure for new modules (utils/, models/)
  - Set up logging configuration with JSON formatting and rotation
  - Create validation utility module with Pydantic schemas
  - Install and configure required dependencies
  - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_

- [ ]* 1.1 Write property test for logging completeness
  - **Property 32: Agent execution logging completeness**
  - **Validates: Requirements 8.1**

- [ ]* 1.2 Write property test for JSON log format
  - **Property 35: JSON log format**
  - **Validates: Requirements 8.5**

- [x] 2. Implement semantic search engine
  - Create SemanticMatcher class in backend/tools/sku_tools.py
  - Implement embedding generation using sentence-transformers (all-MiniLM-L6-v2)
  - Build FAISS index from SKU catalog with persistence to disk
  - Implement similarity search with configurable threshold (default 0.6)
  - Add index rebuild mechanism for catalog updates
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

- [ ]* 2.1 Write property test for embedding dimensionality
  - **Property 1: Embedding dimensionality consistency**
  - **Validates: Requirements 1.1**

- [ ]* 2.2 Write property test for cosine similarity bounds
  - **Property 2: Cosine similarity bounds**
  - **Validates: Requirements 1.2**

- [ ]* 2.3 Write property test for similarity threshold filtering
  - **Property 3: Similarity threshold filtering**
  - **Validates: Requirements 1.3**

- [x] 3. Implement quantity extraction system
  - Create QuantityExtractor class in backend/tools/rfp_tools.py
  - Implement NER-based quantity extraction using spaCy
  - Add regex patterns for quantity formats ("10x", "Quantity: 10", "10 units")
  - Implement fuzzy string matching to map quantities to SKU names
  - Add confidence scoring for extractions
  - Implement fallback to quantity=1 for missing extractions
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

- [ ]* 3.1 Write property test for quantity extraction completeness
  - **Property 4: Quantity extraction completeness**
  - **Validates: Requirements 2.1, 2.2**

- [ ]* 3.2 Write property test for default quantity fallback
  - **Property 6: Default quantity fallback**
  - **Validates: Requirements 2.4**

- [ ]* 3.3 Write property test for confidence score bounds
  - **Property 7: Confidence score bounds**
  - **Validates: Requirements 2.5**

- [x] 4. Implement web scraping and RFP discovery
  - Enhance find_rfp_online() function with real search API integration
  - Implement search query construction with site filters (*.gov, *.org, *.edu)
  - Add BeautifulSoup-based content extraction for HTML
  - Implement PDF text extraction
  - Add timeout enforcement (10 seconds per request)
  - Implement retry logic with exponential backoff
  - Add rate limiting for web requests
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6_

- [ ]* 4.1 Write property test for search query site filters
  - **Property 8: Search query includes site filters**
  - **Validates: Requirements 3.2**

- [ ]* 4.2 Write property test for retry with exponential backoff
  - **Property 9: Retry with exponential backoff**
  - **Validates: Requirements 3.4**

- [ ]* 4.3 Write property test for request timeout enforcement
  - **Property 10: Request timeout enforcement**
  - **Validates: Requirements 3.6**

- [x] 5. Implement structured RFP parsing
  - Create Pydantic models (RFPParsed, RFPRequirement) in backend/models/rfp_models.py
  - Enhance parse_rfp_text() to use LangChain PydanticOutputParser
  - Implement field extraction (title, budget, timeline, requirements, compliance)
  - Add schema validation with error handling
  - Implement partial result return on validation failure
  - _Requirements: 4.1, 4.2, 4.3, 4.4_

- [ ]* 5.1 Write property test for required field extraction
  - **Property 12: Required field extraction**
  - **Validates: Requirements 4.1**

- [ ]* 5.2 Write property test for schema validation enforcement
  - **Property 13: Schema validation enforcement**
  - **Validates: Requirements 4.2**

- [ ]* 5.3 Write property test for Pydantic model return type
  - **Property 15: Pydantic model return type**
  - **Validates: Requirements 4.4**

- [x] 6. Implement inter-agent validation layer
  - Create validation schemas in backend/utils/validation.py
  - Define Pydantic models for each agent output (SalesAgentOutput, TechnicalAgentOutput, PricingAgentOutput)
  - Implement validate_agent_output() function
  - Add validation to Main Agent orchestration between pipeline stages
  - Implement pipeline halt on critical validation failures
  - Add structured error response generation
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

- [ ]* 6.1 Write property test for agent output schema validation
  - **Property 16: Agent output schema validation**
  - **Validates: Requirements 5.1**

- [ ]* 6.2 Write property test for required field presence check
  - **Property 17: Required field presence check**
  - **Validates: Requirements 5.2**

- [ ]* 6.3 Write property test for pipeline halt on critical failure
  - **Property 18: Pipeline halt on critical validation failure**
  - **Validates: Requirements 5.3**

- [ ]* 6.4 Write property test for structured error response
  - **Property 20: Structured error response**
  - **Validates: Requirements 5.5**

- [x] 7. Implement dynamic pricing engine
  - Create PricingStrategy class in backend/tools/sku_tools.py
  - Implement multi-factor margin calculation (complexity, competition, customer type)
  - Add volume discount logic ($50k and $100k thresholds)
  - Enforce margin bounds (15% min, 40% max)
  - Add pricing factors logging for transparency
  - Update estimate_cost() to use dynamic pricing
  - Integrate QuantityExtractor with pricing calculations
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6_

- [ ]* 7.1 Write property test for complexity margin adjustment bounds
  - **Property 21: Complexity margin adjustment bounds**
  - **Validates: Requirements 6.1**

- [ ]* 7.2 Write property test for competition margin adjustment bounds
  - **Property 22: Competition margin adjustment bounds**
  - **Validates: Requirements 6.2**

- [ ]* 7.3 Write property test for customer type margin adjustment bounds
  - **Property 23: Customer type margin adjustment bounds**
  - **Validates: Requirements 6.3**

- [ ]* 7.4 Write property test for volume discount application
  - **Property 24: Volume discount application**
  - **Validates: Requirements 6.4**

- [ ]* 7.5 Write property test for margin bounds enforcement
  - **Property 25: Margin bounds enforcement**
  - **Validates: Requirements 6.5**

- [x] 8. Checkpoint - Ensure core functionality tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 9. Implement caching layer
  - Create CacheManager class in backend/utils/cache.py
  - Implement Redis connection with in-memory fallback
  - Add cache key generation using SHA256 hashing
  - Implement get/set operations with TTL (1 hour default)
  - Create @cached decorator for agent methods
  - Add cache metrics tracking (hits/misses)
  - Implement cache invalidation mechanism
  - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5_

- [ ]* 9.1 Write property test for cache key generation consistency
  - **Property 36: Cache key generation consistency**
  - **Validates: Requirements 9.1**

- [ ]* 9.2 Write property test for cache hit returns stored value
  - **Property 37: Cache hit returns stored value**
  - **Validates: Requirements 9.2**

- [ ]* 9.3 Write property test for cache TTL setting
  - **Property 38: Cache TTL setting**
  - **Validates: Requirements 9.3**

- [ ]* 9.4 Write property test for cache metrics tracking
  - **Property 39: Cache metrics tracking**
  - **Validates: Requirements 9.4**

- [x] 10. Implement retry and error recovery mechanisms
  - Create retry utilities in backend/utils/retry.py
  - Implement @retry_with_backoff decorator using Tenacity
  - Configure exponential backoff (2s min, 10s max, 3 attempts)
  - Add retry attempt logging
  - Implement CircuitBreaker class
  - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5_

- [ ]* 10.1 Write property test for maximum retry attempts
  - **Property 41: Maximum retry attempts**
  - **Validates: Requirements 10.1**

- [ ]* 10.2 Write property test for exponential backoff timing
  - **Property 42: Exponential backoff timing**
  - **Validates: Requirements 10.2**

- [ ]* 10.3 Write property test for circuit breaker activation
  - **Property 45: Circuit breaker activation**
  - **Validates: Requirements 10.5**

- [x] 11. Implement database connection pooling
  - Replace sqlite3 with SQLAlchemy engine in backend/database.py
  - Configure QueuePool (size=10, max_overflow=20, pool_pre_ping=True)
  - Add connection timeout handling
  - Implement proper connection cleanup
  - Add pool metrics endpoint
  - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.5_

- [ ]* 11.1 Write property test for connection health check
  - **Property 48: Connection health check**
  - **Validates: Requirements 12.3**

- [ ]* 11.2 Write property test for connection return to pool
  - **Property 49: Connection return to pool**
  - **Validates: Requirements 12.4**

- [x] 12. Update Technical Agent to use semantic search
  - Replace match_sku() keyword matching with SemanticMatcher
  - Update TechnicalAgent.analyze() to use semantic search
  - Update match confidence scores to use cosine similarity
  - Test with sample RFPs to verify improved accuracy
  - _Requirements: 1.1, 1.2, 1.3_

- [x] 13. Update Pricing Agent to use quantity extraction and dynamic pricing
  - Integrate QuantityExtractor into PricingAgent.analyze()
  - Replace hardcoded quantity dictionary with extracted quantities
  - Update estimate_cost() to use PricingStrategy for dynamic margins
  - Add pricing factors to output for transparency
  - Test with various RFP scenarios
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 6.1, 6.2, 6.3, 6.4, 6.5_

- [x] 14. Update Sales Agent to use web scraping and structured parsing
  - Integrate real web scraping into find_rfp_online()
  - Update parse_rfp_text() to return Pydantic models
  - Add error handling for scraping failures
  - Implement fallback to provided text if scraping fails
  - Test with real RFP searches
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 4.1, 4.2, 4.3, 4.4_

- [x] 15. Checkpoint - Ensure reliability features tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 16. Implement conversation memory system



  - Add LangChain ConversationBufferMemory to MainAgent
  - Implement SQLChatMessageHistory for persistence
  - Add session ID generation and management
  - Update orchestrate() method to store/retrieve conversation context
  - Pass conversation history to agent prompts
  - Implement session memory clearing mechanism
  - Create database table for conversation history
  - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_

- [ ]* 16.1 Write property test for session ID uniqueness
  - **Property 27: Session ID uniqueness**
  - **Validates: Requirements 7.1**

- [ ]* 16.2 Write property test for input-output pair storage
  - **Property 28: Input-output pair storage**
  - **Validates: Requirements 7.2**

- [ ]* 16.3 Write property test for session memory isolation
  - **Property 31: Session memory isolation**
  - **Validates: Requirements 7.5**

- [x] 17. Implement asynchronous agent execution



  - Convert MainAgent.orchestrate() to async function
  - Wrap blocking LLM calls with asyncio.to_thread
  - Implement parallel execution for independent stages using asyncio.gather
  - Update Flask routes to support async handlers
  - Add performance timing comparison logging
  - Test async execution with concurrent requests
  - _Requirements: 11.1, 11.2, 11.3, 11.4_

- [ ]* 17.1 Write property test for async LLM call wrapping
  - **Property 46: Async LLM call wrapping**
  - **Validates: Requirements 11.2**

- [ ]* 17.2 Write property test for performance timing logging
  - **Property 47: Performance timing logging**
  - **Validates: Requirements 11.4**

- [x] 18. Implement API rate limiting


  - Install and configure flask-limiter in backend/app.py
  - Set global rate limit (100 requests/hour per IP)
  - Configure endpoint-specific limits (analyze: 10/min, catalog: 50/min)
  - Use Redis for distributed rate limiting
  - Implement 429 response with retry-after header
  - Add admin bypass mechanism
  - _Requirements: 13.1, 13.2, 13.3, 13.4, 13.5_

- [ ]* 18.1 Write property test for global rate limit enforcement
  - **Property 51: Global rate limit enforcement**
  - **Validates: Requirements 13.1**

- [ ]* 18.2 Write property test for analyze endpoint rate limit
  - **Property 52: Analyze endpoint rate limit**
  - **Validates: Requirements 13.2**

- [ ]* 18.3 Write property test for rate limit response format
  - **Property 54: Rate limit response format**
  - **Validates: Requirements 13.4**


- [x] 19. Implement Prometheus metrics

  - Install prometheus-flask-exporter
  - Expose /metrics endpoint
  - Add request count metrics by endpoint
  - Add response time histogram metrics
  - Add agent execution duration metrics
  - Add error rate metrics by agent
  - Add cache hit/miss ratio metrics
  - Add database connection pool usage metrics
  - _Requirements: 14.1, 14.2, 14.3, 14.4, 14.5, 14.6, 14.7_

- [ ]* 19.1 Write property test for request count recording
  - **Property 56: Request count recording**
  - **Validates: Requirements 14.2**

- [ ]* 19.2 Write property test for response time histogram recording
  - **Property 57: Response time histogram recording**
  - **Validates: Requirements 14.3**

- [ ]* 19.3 Write property test for agent duration recording
  - **Property 58: Agent duration recording**
  - **Validates: Requirements 14.4**

- [x] 20. Implement API authentication



  - Create authentication module in backend/auth.py
  - Implement Bearer token verification using flask-httpauth
  - Create API token storage in environment variables (hashed)
  - Add token verification middleware
  - Protect all /api/* endpoints (exclude /api/health and /metrics)
  - Implement role-based access control (admin/user)
  - Add token expiration (24 hours)
  - Log all authentication attempts
  - Return 401 for invalid tokens, 403 for insufficient permissions
  - _Requirements: 15.1, 15.2, 15.3, 15.4, 15.5, 15.6_

- [ ]* 20.1 Write property test for Bearer token verification
  - **Property 62: Bearer token verification**
  - **Validates: Requirements 15.1**

- [ ]* 20.2 Write property test for invalid token response
  - **Property 63: Invalid token response**
  - **Validates: Requirements 15.3**

- [ ]* 20.3 Write property test for insufficient permissions response
  - **Property 64: Insufficient permissions response**
  - **Validates: Requirements 15.4**

- [ ]* 20.4 Write property test for token expiration setting
  - **Property 65: Token expiration setting**
  - **Validates: Requirements 15.5**

- [x] 21. Update Flask application with production features


  - Add rate limiting to all endpoints
  - Add authentication middleware
  - Add Prometheus metrics export
  - Update routes to support async handlers
  - Add graceful shutdown handling
  - Update error handlers for consistent error responses
  - Add /metrics endpoint for Prometheus
  - Add pool status endpoint at /api/pool-status
  - _Requirements: 13.1, 13.2, 13.3, 13.4, 14.1, 14.2, 15.1, 15.2_



- [ ] 22. Migrate database operations to use connection pool
  - Update all agent initialization to use DatabasePool
  - Replace direct sqlite3 connections with pool connections
  - Update database.py to initialize tables using SQLAlchemy
  - Test connection pool under load


  - Verify proper connection cleanup
  - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.5_

- [ ] 23. Apply retry decorators to external calls
  - Add @retry_with_backoff to LLM API calls in all agents
  - Add @retry_with_backoff to web scraping functions

  - Add @with_circuit_breaker to critical external services
  - Test retry behavior with simulated failures
  - Verify circuit breaker opens after threshold
  - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5_

- [x] 24. Apply caching to agent methods

  - Add @cached decorator to SalesAgent.analyze()
  - Add @cached decorator to TechnicalAgent.analyze()
  - Add @cached decorator to PricingAgent.analyze()
  - Test cache hit/miss behavior
  - Verify TTL expiration
  - Add cache metrics endpoint at /api/cache-metrics
  - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5_

- [x] 25. Create environment configuration and deployment scripts


  - Create .env.example with all required variables
  - Document all environment variables in README
  - Update requirements.txt with all dependencies
  - Add model download script (spaCy, sentence-transformers)
  - Create FAISS index build script
  - Add database migration scripts
  - Document deployment steps
  - _Requirements: All_

- [ ]* 25.1 Write integration tests for complete pipeline
  - Test full RFP processing workflow
  - Test cache hit/miss scenarios
  - Test retry and fallback mechanisms
  - Test authentication and rate limiting
  - Test database connection pooling under load
  - _Requirements: All_


- [x] 26. Final checkpoint - Ensure all tests pass and system is production-ready


  - Ensure all tests pass, ask the user if questions arise.
