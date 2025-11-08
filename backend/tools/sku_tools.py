SKU_TOOLS_PY = """
\"\"\"SKU matching and pricing tools\"\"\"
import json
from typing import List, Dict, Any

def match_sku(rfp_specs: str, db_connection) -> Dict[str, Any]:
    \"\"\"
    Match RFP requirements against SKU catalog
    Returns matched products with confidence scores
    \"\"\"
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

def estimate_cost(matched_skus: List[Dict], db_connection) -> Dict[str, Any]:
    \"\"\"
    Calculate cost estimation based on matched SKUs
    Estimates quantities and applies margins
    \"\"\"
    if not matched_skus:
        return {
            'total_cost': 0,
            'margin': 0,
            'final_price': 0,
            'breakdown': []
        }
    
    # Typical quantities for manufacturing facility RFP
    quantity_estimates = {
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
        quantity = quantity_estimates.get(sku_id, 1)
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
    \"\"\"Get summary statistics of SKU catalog\"\"\"
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
"""

print("=" * 80)
print("FILE STRUCTURE GENERATED - Copy each section to corresponding file")
print("=" * 80)
print("\\n📁 Files to create:\\n")
print("1. requirements.txt")
print("2. .env.example")
print("3. README.md")
print("4. backend/__init__.py")
print("5. backend/database.py")
print("6. backend/tools/__init__.py")
print("7. backend/tools/rfp_tools.py")
print("8. backend/tools/sku_tools.py")
print("\\nContinuing with remaining files...")