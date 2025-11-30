"""Main Agent: Orchestrator for RFP Response Workflow"""
import os
import json
import uuid
import asyncio
import time
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import SQLChatMessageHistory
from backend.utils.validation import validate_pipeline_stage, ValidationError, create_error_response
from backend.utils.logging_config import get_logger

logger = get_logger(__name__)


class MainAgent:
    """
    Main Orchestrator Agent responsible for:
    - Coordinating all sub-agents
    - Managing workflow execution
    - Compiling final RFP response
    - Quality assurance
    - Managing conversation memory
    """
    
    def __init__(self, db_connection, sales_agent, technical_agent, pricing_agent, enable_validation=True, session_id=None):
        self.db_connection = db_connection
        self.sales_agent = sales_agent
        self.technical_agent = technical_agent
        self.pricing_agent = pricing_agent
        self.enable_validation = enable_validation
        self.session_id = session_id or self._generate_session_id()
        self.llm = self._initialize_llm()
        self.memory = self._initialize_memory()
        logger.info(f'Main Agent initialized with validation={enable_validation}, session_id={self.session_id}')
    
    def _generate_session_id(self) -> str:
        """Generate unique session ID"""
        return f"session_{uuid.uuid4().hex[:16]}"
    
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
    
    def _initialize_memory(self):
        """
        Initialize conversation memory with SQL persistence
        
        Returns:
            SQLChatMessageHistory instance
        """
        try:
            # Get database URL from environment or use default
            db_url = os.getenv('DATABASE_URL', 'sqlite:///data/rfp_system.db')
            
            # Create chat message history with SQL persistence
            chat_history = SQLChatMessageHistory(
                session_id=self.session_id,
                connection_string=db_url
            )
            
            logger.info(f'Conversation memory initialized for session {self.session_id}')
            return chat_history
            
        except Exception as e:
            logger.error(f'Failed to initialize conversation memory: {e}')
            # Return basic in-memory history as fallback
            from langchain_core.chat_history import InMemoryChatMessageHistory
            return InMemoryChatMessageHistory()
    
    async def orchestrate(self, rfp_input: str, callback=None) -> dict:
        """
        Orchestrate complete RFP response workflow
        
        Workflow:
        1. Sales Agent: Discover and summarize RFP
        2. Technical Agent: Match products to requirements
        3. Pricing Agent: Calculate costs and pricing
        4. Main Agent: Compile final response
        """
        workflow_result = {
            'timestamp': datetime.now().isoformat(),
            'session_id': self.session_id,
            'rfp_input': rfp_input,
            'stages': {},
            'final_response': {}
        }
        
        # Retrieve conversation context
        try:
            conversation_history = self.memory.messages
            logger.info(f'Retrieved {len(conversation_history)} messages from conversation history')
            workflow_result['conversation_context'] = {
                'message_count': len(conversation_history),
                'has_history': len(conversation_history) > 0
            }
        except Exception as e:
            logger.warning(f'Failed to load conversation history: {e}')
            conversation_history = []
        
        # Track timing for performance comparison
        start_time = time.time()
        stage_times = {}
        
        try:
            # Stage 1: Sales Agent - Discovery & Summary
            if callback:
                callback('sales', 'started')
            
            stage_start = time.time()
            sales_result = await asyncio.to_thread(self.sales_agent.analyze, rfp_input)
            stage_times['sales'] = time.time() - stage_start
            workflow_result['stages']['sales'] = sales_result
            
            # Validate sales agent output
            if self.enable_validation:
                try:
                    validate_pipeline_stage('sales', sales_result)
                    logger.info('Sales agent output validated successfully')
                except ValidationError as e:
                    logger.error(f'Sales agent validation failed: {e.message}')
                    workflow_result['status'] = 'error'
                    workflow_result['error'] = e.message
                    workflow_result['error_details'] = e.to_dict()
                    return workflow_result
            
            if sales_result['status'] == 'error':
                raise Exception(f"Sales Agent failed: {sales_result['error']}")
            
            if callback:
                callback('sales', 'completed', sales_result)
            
            # Stage 2: Technical Agent - SKU Matching
            if callback:
                callback('technical', 'started')
            
            rfp_text = sales_result.get('output', rfp_input)
            stage_start = time.time()
            technical_result = await asyncio.to_thread(self.technical_agent.analyze, rfp_text)
            stage_times['technical'] = time.time() - stage_start
            workflow_result['stages']['technical'] = technical_result
            
            # Validate technical agent output
            if self.enable_validation:
                try:
                    validate_pipeline_stage('technical', technical_result)
                    logger.info('Technical agent output validated successfully')
                except ValidationError as e:
                    logger.error(f'Technical agent validation failed: {e.message}')
                    workflow_result['status'] = 'error'
                    workflow_result['error'] = e.message
                    workflow_result['error_details'] = e.to_dict()
                    return workflow_result
            
            if technical_result['status'] == 'error':
                raise Exception(f"Technical Agent failed: {technical_result['error']}")
            
            if callback:
                callback('technical', 'completed', technical_result)
            
            # Stage 3: Pricing Agent - Cost Estimation
            if callback:
                callback('pricing', 'started')
            
            matched_skus = technical_result.get('matches', [])
            # Pass RFP text for quantity extraction
            stage_start = time.time()
            pricing_result = await asyncio.to_thread(self.pricing_agent.analyze, matched_skus, rfp_text=rfp_text)
            stage_times['pricing'] = time.time() - stage_start
            workflow_result['stages']['pricing'] = pricing_result
            
            # Validate pricing agent output
            if self.enable_validation:
                try:
                    validate_pipeline_stage('pricing', pricing_result)
                    logger.info('Pricing agent output validated successfully')
                except ValidationError as e:
                    logger.error(f'Pricing agent validation failed: {e.message}')
                    workflow_result['status'] = 'error'
                    workflow_result['error'] = e.message
                    workflow_result['error_details'] = e.to_dict()
                    return workflow_result
            
            if pricing_result['status'] == 'error':
                raise Exception(f"Pricing Agent failed: {pricing_result['error']}")
            
            if callback:
                callback('pricing', 'completed', pricing_result)
            
            # Stage 4: Compile Final Response
            if callback:
                callback('main', 'compiling')
            
            final_response = self._compile_final_response(
                sales_result,
                technical_result,
                pricing_result
            )
            
            workflow_result['final_response'] = final_response
            workflow_result['status'] = 'success'
            
            # Calculate total execution time
            total_time = time.time() - start_time
            workflow_result['performance'] = {
                'total_time': round(total_time, 2),
                'stage_times': {k: round(v, 2) for k, v in stage_times.items()},
                'sequential_estimate': round(sum(stage_times.values()), 2),
                'async_benefit': round(sum(stage_times.values()) - total_time, 2) if sum(stage_times.values()) > total_time else 0
            }
            
            # Log performance metrics
            logger.info(
                f'Pipeline completed in {total_time:.2f}s (async)',
                extra={
                    'total_time': total_time,
                    'stage_times': stage_times,
                    'sequential_estimate': sum(stage_times.values()),
                    'execution_mode': 'async'
                }
            )
            
            # Store interaction in conversation memory
            try:
                from langchain_core.messages import HumanMessage, AIMessage
                self.memory.add_message(HumanMessage(content=rfp_input))
                self.memory.add_message(AIMessage(content=json.dumps(final_response)))
                logger.info(f'Saved interaction to conversation memory for session {self.session_id}')
            except Exception as e:
                logger.error(f'Failed to save to conversation memory: {e}')
            
            if callback:
                callback('main', 'completed', final_response)
            
            return workflow_result
            
        except ValidationError as e:
            logger.error(f'Validation error in pipeline: {e.message}')
            workflow_result['status'] = 'error'
            workflow_result['error'] = e.message
            workflow_result['error_details'] = e.to_dict()
            
            if callback:
                callback('main', 'error', e.to_dict())
            
            return workflow_result
        except Exception as e:
            logger.error(f'Unexpected error in pipeline: {e}', exc_info=True)
            workflow_result['status'] = 'error'
            workflow_result['error'] = str(e)
            workflow_result['error_details'] = create_error_response(e, 'main')
            
            if callback:
                callback('main', 'error', {'error': str(e)})
            
            return workflow_result
    
    def _compile_final_response(self, sales_result, technical_result, pricing_result) -> dict:
        """Compile final structured RFP response"""
        return {
            'executive_summary': {
                'project_understanding': sales_result.get('output', ''),
                'proposed_solution': f"We propose {technical_result.get('total_matched', 0)} products matching your requirements",
                'total_investment': pricing_result.get('final_price', 0)
            },
            'technical_proposal': {
                'matched_products': technical_result.get('matches', []),
                'match_confidence': technical_result.get('avg_match_score', 0),
                'compliance': 'All products are UL listed and NEC 2020 compliant'
            },
            'pricing_proposal': {
                'total_cost': pricing_result.get('total_cost', 0),
                'margin': pricing_result.get('margin', 0),
                'final_price': pricing_result.get('final_price', 0),
                'breakdown': pricing_result.get('breakdown', [])
            },
            'delivery_timeline': '6-8 weeks from order confirmation',
            'warranty': '1 year manufacturer warranty on all equipment',
            'next_steps': [
                'Review and approve proposal',
                'Issue purchase order',
                'Schedule delivery and installation',
                'Final inspection and commissioning'
            ]
        }
  
  
    def clear_memory(self):
        """Clear conversation memory for current session"""
        try:
            self.memory.clear()
            logger.info(f'Cleared conversation memory for session {self.session_id}')
        except Exception as e:
            logger.error(f'Failed to clear conversation memory: {e}')
    
    def get_conversation_history(self) -> list:
        """
        Get conversation history for current session
        
        Returns:
            List of message dictionaries
        """
        try:
            messages = self.memory.messages
            
            # Convert messages to dictionaries
            history = []
            for msg in messages:
                history.append({
                    'type': msg.type if hasattr(msg, 'type') else 'unknown',
                    'content': msg.content if hasattr(msg, 'content') else str(msg)
                })
            
            return history
        except Exception as e:
            logger.error(f'Failed to get conversation history: {e}')
            return []
    
    def set_session_id(self, session_id: str):
        """
        Set new session ID and reinitialize memory
        
        Args:
            session_id: New session ID
        """
        self.session_id = session_id
        self.memory = self._initialize_memory()
        logger.info(f'Session ID changed to {session_id}')

    
    def orchestrate_sync(self, rfp_input: str, callback=None) -> dict:
        """
        Synchronous wrapper for orchestrate method
        
        Use this in non-async contexts (e.g., scripts, synchronous Flask routes)
        
        Args:
            rfp_input: RFP input text
            callback: Optional callback function
        
        Returns:
            Workflow result dictionary
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self.orchestrate(rfp_input, callback))
