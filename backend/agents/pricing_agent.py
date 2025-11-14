"""Pricing Agent: Cost Estimation and Proposal Pricing"""
import os
import json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
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
    
    def analyze(self, matched_skus: list) -> dict:
        """Calculate pricing for matched SKUs"""
        try:
            # Get cost estimation
            cost_data = estimate_cost(matched_skus, self.db_connection)
            
            # Use LLM to provide pricing insights
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
            result = chain.invoke({
                "matched_skus": json.dumps(matched_skus, indent=2),
                "cost_breakdown": json.dumps(cost_data, indent=2)
            })
            
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
            return {
                'status': 'error',
                'agent': 'pricing',
                'error': str(e)
            }