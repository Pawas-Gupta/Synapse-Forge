"""Technical Agent: SKU Matching and Specification Analysis"""
import os
import json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
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
    
    def analyze(self, rfp_text: str) -> dict:
        """Analyze RFP and match products"""
        try:
            # Get raw match data
            match_data = match_sku(rfp_text, self.db_connection)
            catalog_summary = get_catalog_summary(self.db_connection)
            
            # Use LLM to analyze and provide insights
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
            result = chain.invoke({
                "rfp_text": rfp_text,
                "matches": json.dumps(match_data['matches'], indent=2),
                "summary": json.dumps(catalog_summary, indent=2)
            })
            
            return {
                'status': 'success',
                'agent': 'technical',
                'output': result.content if hasattr(result, 'content') else str(result),
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