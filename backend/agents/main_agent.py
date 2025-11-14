"""Main Agent: Orchestrator for RFP Response Workflow"""
import os
import json
from datetime import datetime
from langchain_openai import ChatOpenAI


class MainAgent:
    """
    Main Orchestrator Agent responsible for:
    - Coordinating all sub-agents
    - Managing workflow execution
    - Compiling final RFP response
    - Quality assurance
    """
    
    def __init__(self, db_connection, sales_agent, technical_agent, pricing_agent):
        self.db_connection = db_connection
        self.sales_agent = sales_agent
        self.technical_agent = technical_agent
        self.pricing_agent = pricing_agent
        self.llm = self._initialize_llm()
    
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
    
    def orchestrate(self, rfp_input: str, callback=None) -> dict:
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
            'rfp_input': rfp_input,
            'stages': {},
            'final_response': {}
        }
        
        try:
            # Stage 1: Sales Agent - Discovery & Summary
            if callback:
                callback('sales', 'started')
            
            sales_result = self.sales_agent.analyze(rfp_input)
            workflow_result['stages']['sales'] = sales_result
            
            if sales_result['status'] == 'error':
                raise Exception(f"Sales Agent failed: {sales_result['error']}")
            
            if callback:
                callback('sales', 'completed', sales_result)
            
            # Stage 2: Technical Agent - SKU Matching
            if callback:
                callback('technical', 'started')
            
            rfp_text = sales_result.get('output', rfp_input)
            technical_result = self.technical_agent.analyze(rfp_text)
            workflow_result['stages']['technical'] = technical_result
            
            if technical_result['status'] == 'error':
                raise Exception(f"Technical Agent failed: {technical_result['error']}")
            
            if callback:
                callback('technical', 'completed', technical_result)
            
            # Stage 3: Pricing Agent - Cost Estimation
            if callback:
                callback('pricing', 'started')
            
            matched_skus = technical_result.get('matches', [])
            pricing_result = self.pricing_agent.analyze(matched_skus)
            workflow_result['stages']['pricing'] = pricing_result
            
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
            
            if callback:
                callback('main', 'completed', final_response)
            
            return workflow_result
            
        except Exception as e:
            workflow_result['status'] = 'error'
            workflow_result['error'] = str(e)
            
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