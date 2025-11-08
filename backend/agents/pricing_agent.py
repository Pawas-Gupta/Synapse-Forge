"""Pricing Agent: Cost Estimation and Proposal Pricing"""
import os
import json
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from backend.tools.sku_tools import estimate_cost


class PricingAgent:
    """
    Pricing Agent responsible for:
    - Calculating total costs
    - Estimating quantities
    - Applying appropriate margins
    - Generating pricing breakdowns
    """
    
    def __init__(self, db_connection):
        self.db_connection = db_connection
        self.llm = self._initialize_llm()
        self.agent_executor = self._create_agent()
    
    def _initialize_llm(self):
        """Initialize Gemini LLM"""
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        return ChatGoogleGenerativeAI(
            model="gemini-pro",
            google_api_key=api_key,
            temperature=0.1
        )
    
    def _create_agent(self):
        """Create Pricing Agent with tools"""
        def estimate_cost_wrapper(matched_skus_json: str) -> str:
            try:
                matched_skus = json.loads(matched_skus_json)
                result = estimate_cost(matched_skus, self.db_connection)
                return json.dumps(result, indent=2)
            except json.JSONDecodeError:
                return json.dumps({'error': 'Invalid JSON input'})
        
        tools = [
            Tool(
                name="estimate_cost",
                func=estimate_cost_wrapper,
                description=(
                    "Calculate total cost and pricing based on matched SKUs. "
                    "Input: JSON string of matched SKUs array. "
                    "Returns: Cost breakdown with quantities, margins, and final price."
                )
            )
        ]
        
        prompt = PromptTemplate.from_template("""
You are a Pricing Agent specialized in cost estimation and proposal pricing for industrial equipment.

Your responsibilities:
1. Calculate total costs based on matched SKUs
2. Estimate realistic quantities for RFP requirements
3. Apply appropriate profit margins (typically 20-30%)
4. Generate detailed pricing breakdowns
5. Consider volume discounts and project complexity

Available tools: {tools}
Tool names: {tool_names}

When calculating pricing, consider:
- Unit costs from catalog
- Estimated quantities based on project scope
- Standard industry margins
- Competitive positioning
- Project complexity factors

Input: {input}

Pricing analysis:
{agent_scratchpad}
""")
        
        agent = create_react_agent(self.llm, tools, prompt)
        return AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5
        )
    
    def analyze(self, matched_skus: list) -> dict:
        """Calculate pricing for matched SKUs"""
        try:
            # Get cost estimation
            cost_data = estimate_cost(matched_skus, self.db_connection)
            
            # Also run agent for additional insights
            result = self.agent_executor.invoke({
                'input': json.dumps(matched_skus)
            })
            
            return {
                'status': 'success',
                'agent': 'pricing',
                'output': result.get('output', ''),
                'total_cost': cost_data['total_cost'],
                'margin': cost_data['margin'],
                'final_price': cost_data['final_price'],
                'breakdown': cost_data['breakdown']
            }
        except Exception as e:
            return {
                'status': 'error',
                'agent': 'pricing',
                'error': str(e)
            }