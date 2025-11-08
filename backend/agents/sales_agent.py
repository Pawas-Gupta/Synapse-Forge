"""Sales Agent: RFP Discovery and Summarization"""
import os
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from backend.tools.rfp_tools import find_rfp_online, parse_rfp_text


class SalesAgent:
    """
    Sales Agent responsible for:
    - RFP discovery and retrieval
    - High-level requirement extraction
    - Project summarization
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
            temperature=0.3
        )
    
    def _create_agent(self):
        """Create Sales Agent with tools"""
        tools = [
            Tool(
                name="find_rfp_online",
                func=find_rfp_online,
                description=(
                    "Search for and retrieve RFP documents online. "
                    "Input should be a search query string describing the RFP needed."
                )
            ),
            Tool(
                name="parse_rfp",
                func=parse_rfp_text,
                description=(
                    "Parse RFP text and extract structured information. "
                    "Input should be the full RFP text."
                )
            )
        ]
        
        prompt = PromptTemplate.from_template("""
You are a Sales Agent specialized in RFP discovery and summarization for B2B manufacturers.

Your responsibilities:
1. Identify and retrieve relevant RFPs
2. Extract high-level requirements (technical specs, timeline, budget)
3. Summarize project scope and key deliverables
4. Identify potential risks or special requirements

Available tools: {tools}
Tool names: {tool_names}

When analyzing an RFP, focus on:
- Project overview and objectives
- Technical requirements summary
- Timeline and deadlines
- Budget constraints
- Compliance requirements

Input: {input}

Reasoning process:
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
    
    def analyze(self, rfp_input: str) -> dict:
        """Analyze RFP and return summary"""
        try:
            result = self.agent_executor.invoke({
                'input': f"Find and summarize this RFP: {rfp_input}"
            })
            
            return {
                'status': 'success',
                'agent': 'sales',
                'output': result.get('output', ''),
                'intermediate_steps': result.get('intermediate_steps', [])
            }
        except Exception as e:
            return {
                'status': 'error',
                'agent': 'sales',
                'error': str(e)
            }
