"""RFP discovery and processing tools"""
from bs4 import BeautifulSoup
import requests
import os
from typing import List, Optional
from dataclasses import dataclass
from backend.utils.logging_config import get_logger
from backend.utils.retry import retry_with_backoff, with_circuit_breaker
import time

logger = get_logger(__name__)


@dataclass
class RFPDocument:
    """Represents a discovered RFP document"""
    url: str
    title: str
    content: str
    format: str  # 'html', 'pdf', 'doc', 'mock'


def _construct_search_query(query: str) -> str:
    """
    Build search query with site filters
    
    Args:
        query: User search query
    
    Returns:
        Enhanced search query with filters
    """
    # Add RFP-related keywords if not present
    rfp_keywords = ['RFP', 'Request for Proposal', 'tender', 'bid']
    has_rfp_keyword = any(keyword.lower() in query.lower() for keyword in rfp_keywords)
    
    if not has_rfp_keyword:
        query = f"RFP {query}"
    
    # Add site filters for government and organization domains
    site_filter = " (site:*.gov OR site:*.org OR site:*.edu)"
    
    return query + site_filter


@retry_with_backoff(
    max_attempts=3,
    min_wait=2,
    max_wait=10,
    exceptions=(requests.RequestException, requests.Timeout)
)
@with_circuit_breaker('web_scraping', failure_threshold=5, timeout=60)
def _scrape_content(url: str, timeout=10) -> Optional[str]:
    """
    Extract text content from URL
    
    Args:
        url: URL to scrape
        timeout: Request timeout in seconds
    
    Returns:
        Extracted text content or None
    """
    try:
        logger.info(f'Scraping content from {url}')
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        
        # Parse HTML
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text
        text = soup.get_text()
        
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        logger.info(f'Successfully scraped {len(text)} characters')
        return text
        
    except requests.Timeout:
        logger.error(f'Timeout scraping {url}')
        return None
    except requests.RequestException as e:
        logger.error(f'Error scraping {url}: {e}')
        return None
    except Exception as e:
        logger.error(f'Unexpected error scraping {url}: {e}')
        return None


def find_rfp_online(query: str, use_mock=True) -> str:
    """
    Search and scrape RFPs from web
    
    Args:
        query: Search query
        use_mock: If True, return mock data (default for testing)
    
    Returns:
        RFP text content
    """
    if use_mock:
        logger.info(f'Using mock RFP data for query: {query}')
        mock_rfp = f"""
    ========================================
    RFP DOCUMENT
    ========================================
    
    Project Title: Manufacturing Facility Electrical System Upgrade
    RFP Number: RFP-2025-001
    Issue Date: January 15, 2025
    
    PROJECT OVERVIEW:
    Complete electrical system upgrade for 50,000 sq ft manufacturing facility
    located in Chennai, Tamil Nadu. Project includes main distribution upgrades,
    circuit protection, motor controls, and emergency systems.
    
    TECHNICAL REQUIREMENTS:
    
    1. Main Distribution:
       - 1x Distribution panel rated for 400A at 480V
       - Must include 42 circuit spaces
       - Copper bus bars required
       - Main breaker included
    
    2. Circuit Protection:
       - 10x Circuit breakers: 100A capacity, 480V, 3-pole, thermal-magnetic
       - 5x Circuit breakers: 200A capacity, 480V, 3-pole, thermal-magnetic
       - All breakers must be UL listed with 65kA interrupt rating
    
    3. Motor Control:
       - 3x Motor starters for 50HP motors
       - Electronic controls preferred
       - Overload protection required
       - NEMA Size 3 enclosures
    
    4. Power Conversion:
       - 1x Transformer: 75kVA capacity
       - Dry-type construction
       - 480V primary to 208V secondary
       - Minimum 98% efficiency
    
    5. Cable Management:
       - 200 linear feet of 12-inch cable tray system
       - Galvanized steel construction
       - Heavy-duty load rating
       - Pre-galvanized finish
    
    6. Emergency Systems:
       - 15x Emergency lighting units
       - 90-minute battery backup minimum
       - UL924 compliant
       - Self-testing capability preferred
    
    7. Conduit & Raceway:
       - 500 linear feet of 2-inch EMT conduit
       - UL listed
       - Includes couplings and connectors
    
    COMPLIANCE REQUIREMENTS:
    - NEC 2020 compliance mandatory
    - All equipment must be UL listed
    - Installation per manufacturer specifications
    - Final inspection and commissioning required
    
    PROJECT TIMELINE:
    - Proposal Submission Deadline: February 15, 2025
    - Project Start: March 1, 2025
    - Delivery Timeline: 6-8 weeks from award
    - Installation: 4 weeks
    
    BUDGET:
    - Estimated Budget Range: $60,000 - $80,000
    - Payment Terms: Net 30 days
    - Warranty: Minimum 1 year on all equipment
    
    SUBMISSION REQUIREMENTS:
    - Detailed product specifications
    - Itemized pricing breakdown
    - Delivery schedule
    - Warranty information
    - References from similar projects
    
    Contact: procurement@manufacturing-facility.com
    """
        return mock_rfp.strip()
    
    # Real web scraping implementation
    logger.info(f'Searching for RFPs with query: {query}')
    
    # Check for API keys
    serpapi_key = os.getenv('SERPAPI_KEY')
    firecrawl_key = os.getenv('FIRECRAWL_API_KEY')
    
    if serpapi_key:
        # Use SerpAPI for search
        logger.info('Using SerpAPI for RFP discovery')
        try:
            from serpapi import GoogleSearch
            
            search_query = _construct_search_query(query)
            params = {
                "q": search_query,
                "api_key": serpapi_key,
                "num": 5
            }
            
            search = GoogleSearch(params)
            results = search.get_dict()
            
            if 'organic_results' in results and results['organic_results']:
                # Try to scrape the first result
                first_result = results['organic_results'][0]
                url = first_result.get('link')
                
                if url:
                    content = _scrape_content(url)
                    if content:
                        return content
        except ImportError:
            logger.warning('serpapi package not installed')
        except Exception as e:
            logger.error(f'SerpAPI error: {e}')
    
    if firecrawl_key:
        # Use Firecrawl for cleaner extraction
        logger.info('Using Firecrawl for RFP discovery')
        try:
            from firecrawl import FirecrawlApp
            
            app = FirecrawlApp(api_key=firecrawl_key)
            search_query = _construct_search_query(query)
            
            # Search and scrape
            result = app.search(search_query, limit=1)
            if result and len(result) > 0:
                return result[0].get('content', '')
        except ImportError:
            logger.warning('firecrawl package not installed')
        except Exception as e:
            logger.error(f'Firecrawl error: {e}')
    
    # Fallback: Return mock data
    logger.warning('No API keys configured, falling back to mock data')
    return find_rfp_online(query, use_mock=True)

def parse_rfp_text(rfp_text: str, use_llm=False) -> dict:
    """
    Parse RFP text and extract structured information
    
    Args:
        rfp_text: RFP text to parse
        use_llm: If True, use LLM for structured extraction (requires API key)
    
    Returns:
        Dictionary with parsed RFP data (compatible with RFPParsed model)
    """
    logger.info(f'Parsing RFP text ({len(rfp_text)} chars)')
    
    if use_llm:
        # Use LangChain with Pydantic for structured extraction
        try:
            from langchain_openai import ChatOpenAI
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_core.output_parsers import PydanticOutputParser
            from backend.models.rfp_models import RFPParsed
            
            api_key = os.getenv('GROQ_API_KEY')
            if not api_key:
                logger.warning('No GROQ_API_KEY found, falling back to regex parsing')
                return parse_rfp_text(rfp_text, use_llm=False)
            
            llm = ChatOpenAI(
                model="openai/gpt-oss-20b",
                api_key=api_key,
                base_url="https://api.groq.com/openai/v1",
                temperature=0.0
            )
            
            parser = PydanticOutputParser(pydantic_object=RFPParsed)
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an expert at parsing RFP documents and extracting structured information.
Extract all relevant information from the RFP text and format it according to the schema.
{format_instructions}"""),
                ("human", "{rfp_text}")
            ])
            
            chain = prompt | llm | parser
            
            result = chain.invoke({
                "rfp_text": rfp_text,
                "format_instructions": parser.get_format_instructions()
            })
            
            logger.info('Successfully parsed RFP using LLM')
            return result.dict()
            
        except Exception as e:
            logger.error(f'LLM parsing failed: {e}, falling back to regex')
            return parse_rfp_text(rfp_text, use_llm=False)
    
    # Fallback: Regex-based parsing
    logger.info('Using regex-based parsing')
    
    parsed = {
        'title': '',
        'rfp_number': None,
        'budget_min': None,
        'budget_max': None,
        'timeline': None,
        'requirements': [],
        'compliance': [],
        'contact': None,
        'raw_text': rfp_text
    }
    
    # Extract title (first line or "Project Title:" line)
    lines = rfp_text.split('\n')
    for line in lines[:10]:
        if 'project title' in line.lower() or 'title' in line.lower():
            parsed['title'] = line.split(':', 1)[-1].strip()
            break
    if not parsed['title'] and lines:
        parsed['title'] = lines[0].strip()
    
    # Extract RFP number
    import re
    rfp_num_match = re.search(r'RFP\s+Number[:\s]+([A-Z0-9-]+)', rfp_text, re.IGNORECASE)
    if rfp_num_match:
        parsed['rfp_number'] = rfp_num_match.group(1)
    else:
        # Try alternative pattern
        rfp_num_match = re.search(r'RFP[:\s#-]+([A-Z0-9-]+)', rfp_text, re.IGNORECASE)
        if rfp_num_match:
            parsed['rfp_number'] = rfp_num_match.group(1)
    
    # Extract budget
    budget_match = re.search(r'\$\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*(?:-|to)\s*\$\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)', rfp_text)
    if budget_match:
        parsed['budget_min'] = float(budget_match.group(1).replace(',', ''))
        parsed['budget_max'] = float(budget_match.group(2).replace(',', ''))
    
    # Extract timeline
    timeline_match = re.search(r'(?:timeline|deadline|delivery):\s*([^\n]+)', rfp_text, re.IGNORECASE)
    if timeline_match:
        parsed['timeline'] = timeline_match.group(1).strip()
    
    # Extract compliance requirements
    compliance_keywords = ['NEC', 'UL', 'NEMA', 'ISO', 'OSHA', 'compliant', 'compliance']
    for keyword in compliance_keywords:
        if keyword.lower() in rfp_text.lower():
            # Find sentences containing the keyword
            sentences = re.split(r'[.!?]', rfp_text)
            for sentence in sentences:
                if keyword.lower() in sentence.lower():
                    parsed['compliance'].append(sentence.strip())
                    break
    
    # Extract contact
    email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', rfp_text)
    if email_match:
        parsed['contact'] = email_match.group(0)
    
    # Simple keyword extraction for backward compatibility
    parsed['keywords'] = {
        'circuit_breakers': rfp_text.lower().count('circuit breaker'),
        'motor_starters': rfp_text.lower().count('motor starter'),
        'transformer': rfp_text.lower().count('transformer'),
        'distribution': rfp_text.lower().count('distribution'),
        'cable_tray': rfp_text.lower().count('cable tray'),
        'emergency_lighting': rfp_text.lower().count('emergency'),
        'conduit': rfp_text.lower().count('conduit')
    }
    
    parsed['text_length'] = len(rfp_text)
    parsed['has_technical_specs'] = any(term in rfp_text.lower() for term in ['voltage', 'current', 'amp'])
    
    logger.info(f'Parsed RFP: title="{parsed["title"]}", budget={parsed["budget_min"]}-{parsed["budget_max"]}')
    
    return parsed


# Quantity Extraction Implementation
import re
import spacy
from typing import Dict, List, Tuple, Optional
from fuzzywuzzy import fuzz
from dataclasses import dataclass
from backend.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class QuantityInfo:
    """Information about an extracted quantity"""
    quantity: int
    confidence: float
    context: str
    extraction_method: str  # 'pattern', 'ner', or 'default'


class QuantityExtractor:
    """
    Extract quantities from RFP text using NER and regex patterns
    
    Uses spaCy for Named Entity Recognition and regex for pattern matching.
    Maps extracted quantities to SKU names using fuzzy string matching.
    """
    
    def __init__(self, spacy_model='en_core_web_sm'):
        """
        Initialize quantity extractor
        
        Args:
            spacy_model: spaCy model name (default: en_core_web_sm)
        """
        logger.info(f'Loading spaCy model: {spacy_model}')
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            logger.error(f'spaCy model {spacy_model} not found. Run: python -m spacy download {spacy_model}')
            raise
        
        # Regex patterns for quantity extraction
        self.patterns = [
            # Pattern: "10x", "5X"
            (r'(\d+)\s*[xX]\s+([^,\n\.]+)', 'multiplier'),
            # Pattern: "Quantity: 10", "Qty: 5"
            (r'(?:quantity|qty)[\s:]+(\d+)(?:\s+([^,\n\.]+))?', 'explicit'),
            # Pattern: "10 units", "5 pieces"
            (r'(\d+)\s+(?:units?|pieces?|items?)\s+(?:of\s+)?([^,\n\.]+)', 'units'),
            # Pattern: "200 linear feet of"
            (r'(\d+)\s+(?:linear\s+)?(?:feet|ft|meters?|m)\s+(?:of\s+)?([^,\n\.]+)', 'measurement'),
            # Pattern: "- 10x" or "• 10x" (bullet points)
            (r'[-•]\s*(\d+)\s*[xX]?\s+([^,\n\.]+)', 'bullet'),
        ]
    
    def _parse_patterns(self, text: str) -> List[Tuple[int, str, str]]:
        """
        Extract (quantity, context, method) using regex patterns
        
        Args:
            text: Input text
        
        Returns:
            List of (quantity, context, method) tuples
        """
        extractions = []
        
        for pattern, method in self.patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    quantity = int(match.group(1))
                    context = match.group(2) if len(match.groups()) > 1 and match.group(2) else ""
                    context = context.strip() if context else ""
                    # Limit context length
                    context = context[:100]
                    extractions.append((quantity, context, method))
                except (ValueError, IndexError, AttributeError):
                    continue
        
        return extractions
    
    def _extract_with_ner(self, text: str) -> List[Tuple[int, str, str]]:
        """
        Extract quantities using spaCy NER
        
        Args:
            text: Input text
        
        Returns:
            List of (quantity, context, method) tuples
        """
        extractions = []
        doc = self.nlp(text)
        
        # Look for CARDINAL entities (numbers)
        for ent in doc.ents:
            if ent.label_ == 'CARDINAL':
                try:
                    quantity = int(ent.text.replace(',', ''))
                    # Get surrounding context (5 words before and after)
                    start_idx = max(0, ent.start - 5)
                    end_idx = min(len(doc), ent.end + 5)
                    context = doc[start_idx:end_idx].text
                    extractions.append((quantity, context, 'ner'))
                except ValueError:
                    continue
        
        return extractions
    
    def _fuzzy_match_sku(self, context: str, sku_names: List[str]) -> Tuple[Optional[str], float]:
        """
        Match context to SKU name using fuzzy string matching
        
        Args:
            context: Context text containing potential SKU reference
            sku_names: List of SKU names to match against
        
        Returns:
            Tuple of (best_match_sku_name, confidence_score)
        """
        if not sku_names:
            return None, 0.0
        
        best_match = None
        best_score = 0.0
        
        context_lower = context.lower()
        
        for sku_name in sku_names:
            # Calculate fuzzy match score
            score = fuzz.partial_ratio(context_lower, sku_name.lower())
            
            if score > best_score:
                best_score = score
                best_match = sku_name
        
        # Convert score to 0-1 range
        confidence = best_score / 100.0
        
        return best_match, confidence
    
    def extract(self, rfp_text: str, sku_names: List[str]) -> Dict[str, QuantityInfo]:
        """
        Extract quantities and map to SKUs
        
        Args:
            rfp_text: RFP specification text
            sku_names: List of SKU names to match against
        
        Returns:
            Dictionary mapping SKU names to QuantityInfo objects
        """
        logger.info(f'Extracting quantities from text ({len(rfp_text)} chars) for {len(sku_names)} SKUs')
        
        # Extract using both methods
        pattern_extractions = self._parse_patterns(rfp_text)
        ner_extractions = self._extract_with_ner(rfp_text)
        
        all_extractions = pattern_extractions + ner_extractions
        logger.info(f'Found {len(all_extractions)} quantity mentions')
        
        # Map extractions to SKUs
        sku_quantities = {}
        
        for quantity, context, method in all_extractions:
            # Find best matching SKU
            matched_sku, match_confidence = self._fuzzy_match_sku(context, sku_names)
            
            if matched_sku and match_confidence >= 0.6:  # Threshold for fuzzy matching
                # If SKU already has a quantity, keep the one with higher confidence
                if matched_sku in sku_quantities:
                    if match_confidence > sku_quantities[matched_sku].confidence:
                        sku_quantities[matched_sku] = QuantityInfo(
                            quantity=quantity,
                            confidence=match_confidence,
                            context=context,
                            extraction_method=method
                        )
                else:
                    sku_quantities[matched_sku] = QuantityInfo(
                        quantity=quantity,
                        confidence=match_confidence,
                        context=context,
                        extraction_method=method
                    )
        
        # Add default quantity=1 for SKUs without extracted quantities
        for sku_name in sku_names:
            if sku_name not in sku_quantities:
                sku_quantities[sku_name] = QuantityInfo(
                    quantity=1,
                    confidence=0.0,
                    context="",
                    extraction_method='default'
                )
        
        logger.info(f'Mapped quantities to {len([q for q in sku_quantities.values() if q.extraction_method != "default"])} SKUs')
        
        return sku_quantities


def extract_quantities_from_rfp(rfp_text: str, sku_list: List[Dict]) -> Dict[str, int]:
    """
    Extract quantities from RFP text and map to SKUs
    
    Args:
        rfp_text: RFP specification text
        sku_list: List of SKU dictionaries with 'name' field
    
    Returns:
        Dictionary mapping SKU IDs to quantities
    """
    # Extract SKU names
    sku_names = [sku['name'] for sku in sku_list]
    sku_id_to_name = {sku['sku_id']: sku['name'] for sku in sku_list}
    
    # Extract quantities
    extractor = QuantityExtractor()
    quantity_info = extractor.extract(rfp_text, sku_names)
    
    # Map back to SKU IDs
    result = {}
    for sku_id, sku_name in sku_id_to_name.items():
        if sku_name in quantity_info:
            result[sku_id] = quantity_info[sku_name].quantity
        else:
            result[sku_id] = 1  # Default
    
    return result
