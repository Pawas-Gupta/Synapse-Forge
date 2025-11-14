"""Sales Agent: RFP Discovery and Summarization"""
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
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
    
    def analyze(self, rfp_input: str) -> dict:
        """Analyze RFP and return summary"""
        try:
            # First, try to find RFP online if it looks like a query
            if len(rfp_input) < 200:
                rfp_text = find_rfp_online(rfp_input)
            else:
                rfp_text = rfp_input
            
            # Parse the RFP
            parsed_data = parse_rfp_text(rfp_text)
            
            # Use LLM to summarize
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
            result = chain.invoke({"rfp_text": rfp_text})
            
            return {
                'status': 'success',
                'agent': 'sales',
                'output': result.content if hasattr(result, 'content') else str(result),
                'parsed_data': parsed_data
            }
        except Exception as e:
            return {
                'status': 'error',
                'agent': 'sales',
                'error': str(e)
            }