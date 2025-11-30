#SKU matching and pricing tools
import json
from typing import List, Dict, Any

def match_sku(rfp_specs: str, db_connection) -> Dict[str, Any]:
    """
    Match RFP requirements against SKU catalog
    Returns matched products with confidence scores
    """
    cursor = db_connection.cursor()
    
    # Extract keywords from RFP
    keywords = {
        'circuit breaker': ['100A', '200A'],
        'motor starter': ['50HP'],
        'transformer': ['75kVA'],
        'distribution panel': ['400A'],
        'cable tray': ['12in'],
        'emergency lighting': ['battery'],
        'conduit': ['2in', 'EMT']
    }
    
    matches = []
    
    for keyword, variants in keywords.items():
        # Search in name or description
        cursor.execute('''
            SELECT * FROM SKU_CATALOG 
            WHERE LOWER(name) LIKE ? OR LOWER(description) LIKE ?
        ''', (f'%{keyword}%', f'%{keyword}%'))
        
        results = cursor.fetchall()
        
        for row in results:
            sku_id, name, description, specs_json, unit_cost, category, created_at = row
            specs = json.loads(specs_json)
            
            # Calculate match score based on keyword presence
            base_score = 75
            if any(variant.lower() in name.lower() for variant in variants):
                base_score += 15
            if any(variant.lower() in str(specs) for variant in variants):
                base_score += 10
            
            match_score = min(base_score, 98)  # Cap at 98%
            
            matches.append({
                'sku_id': sku_id,
                'name': name,
                'description': description,
                'specs': specs,
                'unit_cost': unit_cost,
                'category': category,
                'match_score': match_score
            })
    
    # Remove duplicates
    unique_matches = {m['sku_id']: m for m in matches}.values()
    final_matches = list(unique_matches)
    
    avg_score = sum(m['match_score'] for m in final_matches) / len(final_matches) if final_matches else 0
    
    return {
        'matches': final_matches,
        'avg_match_score': round(avg_score, 2),
        'total_items_matched': len(final_matches)
    }

def estimate_cost(matched_skus: List[Dict], db_connection, quantities: Dict[str, int] = None) -> Dict[str, Any]:
    
    """Calculate cost estimation based on matched SKUs
    Estimates quantities and applies margins
    
    Args:
        matched_skus: List of matched SKU dictionaries
        db_connection: Database connection
        quantities: Optional dict mapping SKU IDs to quantities (from extraction)
    """
    
    if not matched_skus:
        return {
            'total_cost': 0,
            'margin': 0,
            'final_price': 0,
            'breakdown': []
        }
    
    # Default quantities for manufacturing facility RFP (fallback)
    default_quantity_estimates = {
        'SKU-001': 10,   # Circuit breakers 100A
        'SKU-002': 5,    # Circuit breakers 200A
        'SKU-003': 3,    # Motor starters 50HP
        'SKU-004': 1,    # Distribution panel 400A
        'SKU-005': 1,    # Transformer 75kVA
        'SKU-006': 20,   # Cable tray (200ft / 10ft per unit)
        'SKU-007': 15,   # Emergency lights
        'SKU-008': 50    # Conduit (500ft / 10ft per unit)
    }
    
    breakdown = []
    total_cost = 0
    
    for sku in matched_skus:
        sku_id = sku['sku_id']
        
        # Use extracted quantities if available, otherwise use defaults
        if quantities and sku_id in quantities:
            quantity = quantities[sku_id]
        else:
            quantity = default_quantity_estimates.get(sku_id, 1)
        
        unit_cost = sku['unit_cost']
        line_total = quantity * unit_cost
        
        breakdown.append({
            'sku_id': sku_id,
            'name': sku['name'],
            'category': sku.get('category', 'General'),
            'quantity': quantity,
            'unit_cost': round(unit_cost, 2),
            'line_total': round(line_total, 2)
        })
        
        total_cost += line_total
    
    # Apply 25% margin
    margin_percent = 0.25
    margin = total_cost * margin_percent
    final_price = total_cost + margin
    
    return {
        'total_cost': round(total_cost, 2),
        'margin': round(margin, 2),
        'margin_percent': margin_percent * 100,
        'final_price': round(final_price, 2),
        'breakdown': breakdown,
        'item_count': len(breakdown)
    }

def get_catalog_summary(db_connection) -> Dict[str, Any]:
    """Get summary statistics of SKU catalog"""
    cursor = db_connection.cursor()
    
    cursor.execute('SELECT COUNT(*) FROM SKU_CATALOG')
    total_skus = cursor.fetchone()[0]
    
    cursor.execute('SELECT category, COUNT(*) FROM SKU_CATALOG GROUP BY category')
    categories = dict(cursor.fetchall())
    
    cursor.execute('SELECT AVG(unit_cost), MIN(unit_cost), MAX(unit_cost) FROM SKU_CATALOG')
    avg_cost, min_cost, max_cost = cursor.fetchone()
    
    return {
        'total_skus': total_skus,
        'categories': categories,
        'price_stats': {
            'average': round(avg_cost, 2),
            'minimum': round(min_cost, 2),
            'maximum': round(max_cost, 2)
        }
    }


# Semantic Search Implementation
import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from typing import Optional, Tuple
from backend.utils.logging_config import get_logger

logger = get_logger(__name__)


class SemanticMatcher:
    """
    Semantic search engine for SKU matching using sentence-transformers and FAISS
    
    Uses all-MiniLM-L6-v2 model for generating 384-dimensional embeddings
    and FAISS for efficient similarity search.
    """
    
    def __init__(self, db_connection, model_name='all-MiniLM-L6-v2', index_path='data/faiss_index.bin'):
        """
        Initialize semantic matcher
        
        Args:
            db_connection: Database connection
            model_name: Sentence transformer model name (default: all-MiniLM-L6-v2)
            index_path: Path to save/load FAISS index
        """
        self.db_connection = db_connection
        self.model_name = model_name
        self.index_path = index_path
        self.metadata_path = index_path.replace('.bin', '_metadata.pkl')
        
        # Initialize model
        logger.info(f'Loading sentence transformer model: {model_name}')
        self.model = SentenceTransformer(model_name)
        
        # Initialize index and metadata
        self.index = None
        self.sku_metadata = []  # List of (sku_id, name, description, specs, unit_cost, category)
        
        # Load or build index
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            self.load_index()
        else:
            logger.info('FAISS index not found, building new index')
            self.build_index()
    
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding vector for text
        
        Args:
            text: Input text
        
        Returns:
            Normalized embedding vector (384 dimensions)
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        # Normalize for cosine similarity
        embedding = embedding / np.linalg.norm(embedding)
        return embedding
    
    def build_index(self, force_rebuild=False) -> None:
        """
        Build FAISS index from SKU catalog
        
        Args:
            force_rebuild: Force rebuild even if index exists
        """
        if not force_rebuild and self.index is not None:
            logger.info('Index already built, skipping')
            return
        
        logger.info('Building FAISS index from SKU catalog')
        
        # Fetch all SKUs from database
        cursor = self.db_connection.cursor()
        cursor.execute('SELECT sku_id, name, description, specs, unit_cost, category FROM SKU_CATALOG')
        skus = cursor.fetchall()
        
        if not skus:
            logger.warning('No SKUs found in catalog')
            return
        
        # Generate embeddings for all SKUs
        embeddings = []
        self.sku_metadata = []
        
        for sku in skus:
            sku_id, name, description, specs_json, unit_cost, category = sku
            specs = json.loads(specs_json)
            
            # Combine name, description, and specs for embedding
            text = f"{name}. {description}. Specifications: {json.dumps(specs)}"
            embedding = self.get_embedding(text)
            
            embeddings.append(embedding)
            self.sku_metadata.append({
                'sku_id': sku_id,
                'name': name,
                'description': description,
                'specs': specs,
                'unit_cost': unit_cost,
                'category': category
            })
        
        # Create FAISS index (IndexFlatIP for cosine similarity with normalized vectors)
        embeddings_array = np.array(embeddings).astype('float32')
        dimension = embeddings_array.shape[1]
        
        self.index = faiss.IndexFlatIP(dimension)  # Inner product = cosine similarity for normalized vectors
        self.index.add(embeddings_array)
        
        logger.info(f'Built FAISS index with {len(embeddings)} SKUs, dimension={dimension}')
        
        # Save index and metadata
        self.save_index()
    
    def save_index(self) -> None:
        """Save FAISS index and metadata to disk"""
        # Create data directory if it doesn't exist
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, self.index_path)
        
        # Save metadata
        with open(self.metadata_path, 'wb') as f:
            pickle.dump(self.sku_metadata, f)
        
        logger.info(f'Saved FAISS index to {self.index_path}')
    
    def load_index(self) -> None:
        """Load FAISS index and metadata from disk"""
        try:
            self.index = faiss.read_index(self.index_path)
            
            with open(self.metadata_path, 'rb') as f:
                self.sku_metadata = pickle.load(f)
            
            logger.info(f'Loaded FAISS index from {self.index_path} with {len(self.sku_metadata)} SKUs')
        except Exception as e:
            logger.error(f'Failed to load FAISS index: {e}')
            logger.info('Building new index')
            self.build_index()
    
    def match(self, rfp_text: str, top_k=10, threshold=0.6) -> List[Dict[str, Any]]:
        """
        Find semantically similar SKUs
        
        Args:
            rfp_text: RFP specification text
            top_k: Number of top matches to return
            threshold: Minimum similarity score (0-1)
        
        Returns:
            List of match dictionaries with sku_id, score, and metadata
        """
        if self.index is None or len(self.sku_metadata) == 0:
            logger.warning('Index not built, building now')
            self.build_index()
        
        # Generate embedding for RFP text
        query_embedding = self.get_embedding(rfp_text)
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        
        # Search FAISS index
        scores, indices = self.index.search(query_embedding, min(top_k, len(self.sku_metadata)))
        
        # Filter by threshold and format results
        matches = []
        for score, idx in zip(scores[0], indices[0]):
            # Convert inner product back to similarity score (0-1 range)
            similarity = float(score)
            
            if similarity >= threshold:
                metadata = self.sku_metadata[idx]
                matches.append({
                    'sku_id': metadata['sku_id'],
                    'name': metadata['name'],
                    'description': metadata['description'],
                    'specs': metadata['specs'],
                    'unit_cost': metadata['unit_cost'],
                    'category': metadata['category'],
                    'match_score': round(similarity * 100, 2)  # Convert to percentage
                })
        
        logger.info(f'Found {len(matches)} matches above threshold {threshold}')
        return matches


def match_sku_semantic(rfp_specs: str, db_connection, top_k=10, threshold=0.6) -> Dict[str, Any]:
    """
    Match RFP requirements against SKU catalog using semantic search
    
    Args:
        rfp_specs: RFP specification text
        db_connection: Database connection
        top_k: Number of top matches to return
        threshold: Minimum similarity score (0-1)
    
    Returns:
        Dictionary with matches, avg_match_score, and total_items_matched
    """
    matcher = SemanticMatcher(db_connection)
    matches = matcher.match(rfp_specs, top_k=top_k, threshold=threshold)
    
    avg_score = sum(m['match_score'] for m in matches) / len(matches) if matches else 0
    
    return {
        'matches': matches,
        'avg_match_score': round(avg_score, 2),
        'total_items_matched': len(matches)
    }



# Dynamic Pricing Engine
class PricingStrategy:
    """
    Multi-factor pricing algorithm for dynamic margin calculation
    
    Adjusts margins based on:
    - Project complexity (simple/medium/complex)
    - Competition level (high/medium/low)
    - Customer type (new/returning/enterprise)
    - Order volume (discounts for >$50k, >$100k)
    """
    
    def __init__(self, base_margin=0.25, min_margin=0.15, max_margin=0.40):
        """
        Initialize pricing strategy
        
        Args:
            base_margin: Base margin percentage (default: 0.25 = 25%)
            min_margin: Minimum allowed margin (default: 0.15 = 15%)
            max_margin: Maximum allowed margin (default: 0.40 = 40%)
        """
        self.base_margin = base_margin
        self.min_margin = min_margin
        self.max_margin = max_margin
        self.last_factors = {}
        
        logger.info(f'PricingStrategy initialized: base={base_margin}, min={min_margin}, max={max_margin}')
    
    def calculate_margin(self,
                        total_cost: float,
                        complexity: str = 'medium',
                        competition: str = 'medium',
                        customer_type: str = 'new') -> float:
        """
        Calculate dynamic margin percentage
        
        Args:
            total_cost: Total cost before margin
            complexity: Project complexity ('simple', 'medium', 'complex')
            competition: Competition level ('high', 'medium', 'low')
            customer_type: Customer type ('new', 'returning', 'enterprise')
        
        Returns:
            Margin as decimal (e.g., 0.28 for 28%)
        """
        # Start with base margin
        margin = self.base_margin
        
        # Complexity adjustment
        complexity_adj = {
            'simple': -0.05,
            'medium': 0.00,
            'complex': +0.10
        }
        complexity_factor = complexity_adj.get(complexity.lower(), 0.00)
        margin += complexity_factor
        
        # Competition adjustment
        competition_adj = {
            'high': -0.08,
            'medium': 0.00,
            'low': +0.12
        }
        competition_factor = competition_adj.get(competition.lower(), 0.00)
        margin += competition_factor
        
        # Customer type adjustment
        customer_adj = {
            'new': 0.00,
            'returning': -0.03,
            'enterprise': +0.08
        }
        customer_factor = customer_adj.get(customer_type.lower(), 0.00)
        margin += customer_factor
        
        # Volume discount
        volume_factor = self.apply_volume_discount(total_cost, margin)
        margin = volume_factor
        
        # Enforce bounds
        margin = max(self.min_margin, min(margin, self.max_margin))
        
        # Store factors for transparency
        self.last_factors = {
            'base_margin': self.base_margin,
            'complexity': complexity,
            'complexity_adjustment': complexity_factor,
            'competition': competition,
            'competition_adjustment': competition_factor,
            'customer_type': customer_type,
            'customer_adjustment': customer_factor,
            'volume_discount_applied': volume_factor != (self.base_margin + complexity_factor + competition_factor + customer_factor),
            'final_margin': margin,
            'bounded': margin != (self.base_margin + complexity_factor + competition_factor + customer_factor + (volume_factor - (self.base_margin + complexity_factor + competition_factor + customer_factor)))
        }
        
        logger.info(f'Calculated margin: {margin:.2%} for cost=${total_cost:.2f}', extra=self.last_factors)
        
        return margin
    
    def apply_volume_discount(self, total_cost: float, current_margin: float) -> float:
        """
        Apply volume-based margin adjustments
        
        Args:
            total_cost: Total cost before margin
            current_margin: Current margin before volume adjustment
        
        Returns:
            Adjusted margin
        """
        if total_cost > 100000:
            return current_margin - 0.10
        elif total_cost > 50000:
            return current_margin - 0.05
        else:
            return current_margin
    
    def get_pricing_factors(self) -> dict:
        """
        Return dict of factors used in last calculation
        
        Returns:
            Dictionary with all pricing factors
        """
        return self.last_factors.copy()


def estimate_cost_with_dynamic_pricing(
    matched_skus: List[Dict],
    db_connection,
    quantities: Dict[str, int] = None,
    complexity: str = 'medium',
    competition: str = 'medium',
    customer_type: str = 'new'
) -> Dict[str, Any]:
    """
    Calculate cost estimation with dynamic pricing
    
    Args:
        matched_skus: List of matched SKU dictionaries
        db_connection: Database connection
        quantities: Optional dict mapping SKU IDs to quantities
        complexity: Project complexity
        competition: Competition level
        customer_type: Customer type
    
    Returns:
        Cost breakdown with dynamic margin
    """
    # First calculate base cost
    base_result = estimate_cost(matched_skus, db_connection, quantities)
    
    if base_result['total_cost'] == 0:
        return base_result
    
    # Apply dynamic pricing
    pricing_strategy = PricingStrategy()
    dynamic_margin_pct = pricing_strategy.calculate_margin(
        base_result['total_cost'],
        complexity=complexity,
        competition=competition,
        customer_type=customer_type
    )
    
    # Recalculate with dynamic margin
    dynamic_margin = base_result['total_cost'] * dynamic_margin_pct
    dynamic_final_price = base_result['total_cost'] + dynamic_margin
    
    return {
        'total_cost': base_result['total_cost'],
        'margin': round(dynamic_margin, 2),
        'margin_percent': round(dynamic_margin_pct * 100, 2),
        'final_price': round(dynamic_final_price, 2),
        'breakdown': base_result['breakdown'],
        'item_count': base_result['item_count'],
        'pricing_factors': pricing_strategy.get_pricing_factors()
    }
