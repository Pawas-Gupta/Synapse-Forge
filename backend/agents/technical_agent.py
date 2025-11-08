"""Technical Agent: SKU Matching and Specification Analysis"""
import os
import json
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from backend.tools.sku_tools import match_sku, get_catalog_summary


class TechnicalAgent:
    """
    Technical Agent responsible for:
    - Analyzing technical specifications
    - Matching requirements to SKU catalog
    - Calculating specification match scores
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
            temperature=0.2
        )
    
    def _create_agent(self):
        """Create Technical Agent with tools"""
        def match_sku_wrapper(rfp_specs: str) -> str:
            result = match_sku(rfp_specs, self.db_connection)
            return json.dumps(result, indent=2)
        
        def catalog_summary_wrapper(query: str) -> str:
            summary = get_catalog_summary(self.db_connection)
            return json.dumps(summary, indent=2)
        
        tools = [
            Tool(
                name="match_sku",
                func=match_sku_wrapper,
                description=(
                    "Match RFP technical requirements against SKU catalog. "
                    "Input: RFP specifications text. "
                    "Returns: Matched SKUs with confidence scores as JSON."
                )
            ),
            Tool(
                name="catalog_summary",
                func=catalog_summary_wrapper,
                description=(
                    "Get summary of available SKU catalog including categories and pricing. "
                    "Input: Any query string. "
                    "Returns: Catalog statistics as JSON."
                )
            )
        ]
        
        prompt = PromptTemplate.from_template("""
You are a Technical Agent specialized in product matching and specification analysis for industrial equipment.

Your responsibilities:
1. Analyze technical requirements from RFPs
2. Match requirements against SKU catalog database
3. Calculate specification match confidence scores
4. Identify best-fit products for each requirement
5. Flag any requirements that cannot be met

Available tools: {tools}
Tool names: {tool_names}

When matching products, consider:
- Voltage and current ratings
- Physical specifications
- Compliance requirements (UL, NEC)
- Performance characteristics
- Compatibility with existing systems

Input: {input}

Technical analysis:
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
    
    def analyze(self, rfp_text: str) -> dict:
        """Analyze RFP and match products"""
        try:
            result = self.agent_executor.invoke({
                'input': f"Match products for this RFP: {rfp_text}"
            })
            
            # Also get raw match data
            match_data = match_sku(rfp_text, self.db_connection)
            
            return {
                'status': 'success',
                'agent': 'technical',
                'output': result.get('output', ''),
                'matches': match_data['matches'],
                'avg_match_score': match_data['avg_match_score'],
                'total_matched': match_data['total_items_matched']
            }
        except Exception as e:
            return {
                'status': 'error',
                'agent': 'technical',
                'error': str(e)
            }