"""Flask REST API Backend for RFP Agent System"""
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os
import json

from backend.database import get_db
from backend.agents.main_agent import MainAgent
from backend.agents.sales_agent import SalesAgent
from backend.agents.technical_agent import TechnicalAgent
from backend.agents.pricing_agent import PricingAgent
from backend.tools.sku_tools import match_sku, estimate_cost, get_catalog_summary

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global variables for agents
db = None
agents = {}


def initialize_agents():
    """Initialize all agents"""
    global db, agents
    
    if db is None:
        db = get_db()
    
    if not agents:
        db_conn = db.get_connection()
        
        agents['sales'] = SalesAgent(db_conn)
        agents['technical'] = TechnicalAgent(db_conn)
        agents['pricing'] = PricingAgent(db_conn)
        agents['main'] = MainAgent(
            db_conn,
            agents['sales'],
            agents['technical'],
            agents['pricing']
        )


# ============================================================================
# API ROUTES
# ============================================================================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'RFP Agent System API',
        'version': '1.0.0'
    })


@app.route('/api/catalog', methods=['GET'])
def get_catalog():
    """Get SKU catalog"""
    try:
        initialize_agents()
        db_conn = db.get_connection()
        
        # Get all SKUs
        cursor = db_conn.cursor()
        cursor.execute('SELECT * FROM SKU_CATALOG')
        rows = cursor.fetchall()
        
        catalog = []
        for row in rows:
            sku_id, name, description, specs, unit_cost, category, created_at = row
            catalog.append({
                'sku_id': sku_id,
                'name': name,
                'description': description,
                'specs': json.loads(specs),
                'unit_cost': unit_cost,
                'category': category
            })
        
        # Get summary
        summary = get_catalog_summary(db_conn)
        
        return jsonify({
            'status': 'success',
            'catalog': catalog,
            'summary': summary
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500


@app.route('/api/match', methods=['POST'])
def match_products():
    """Match products to RFP requirements"""
    try:
        data = request.get_json()
        rfp_text = data.get('rfp_text', '')
        
        if not rfp_text:
            return jsonify({
                'status': 'error',
                'error': 'rfp_text is required'
            }), 400
        
        initialize_agents()
        db_conn = db.get_connection()
        
        # Perform matching
        match_result = match_sku(rfp_text, db_conn)
        
        return jsonify({
            'status': 'success',
            'matches': match_result['matches'],
            'avg_match_score': match_result['avg_match_score'],
            'total_matched': match_result['total_items_matched']
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500


@app.route('/api/pricing', methods=['POST'])
def calculate_pricing():
    """Calculate pricing for matched SKUs"""
    try:
        data = request.get_json()
        matched_skus = data.get('matched_skus', [])
        
        if not matched_skus:
            return jsonify({
                'status': 'error',
                'error': 'matched_skus array is required'
            }), 400
        
        initialize_agents()
        db_conn = db.get_connection()
        
        # Calculate pricing
        pricing_result = estimate_cost(matched_skus, db_conn)
        
        return jsonify({
            'status': 'success',
            'pricing': pricing_result
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500


@app.route('/api/rfp/analyze', methods=['POST'])
def analyze_rfp():
    """Analyze complete RFP using all agents"""
    try:
        data = request.get_json()
        rfp_input = data.get('rfp_input', '')
        
        if not rfp_input:
            return jsonify({
                'status': 'error',
                'error': 'rfp_input is required'
            }), 400
        
        initialize_agents()
        
        # Orchestrate workflow
        result = agents['main'].orchestrate(rfp_input)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500


@app.route('/api/agents/sales', methods=['POST'])
def sales_agent_analyze():
    """Run Sales Agent analysis"""
    try:
        data = request.get_json()
        rfp_input = data.get('rfp_input', '')
        
        initialize_agents()
        result = agents['sales'].analyze(rfp_input)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500


@app.route('/api/agents/technical', methods=['POST'])
def technical_agent_analyze():
    """Run Technical Agent analysis"""
    try:
        data = request.get_json()
        rfp_text = data.get('rfp_text', '')
        
        initialize_agents()
        result = agents['technical'].analyze(rfp_text)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500


@app.route('/api/agents/pricing', methods=['POST'])
def pricing_agent_analyze():
    """Run Pricing Agent analysis"""
    try:
        data = request.get_json()
        matched_skus = data.get('matched_skus', [])
        
        initialize_agents()
        result = agents['pricing'].analyze(matched_skus)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500


# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == '__main__':
    host = os.getenv('FLASK_HOST', '127.0.0.1')
    port = int(os.getenv('FLASK_PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'
    
    print("=" * 80)
    print("🚀 RFP Agent System - Flask Backend")
    print("=" * 80)
    print(f"Server running on: http://{host}:{port}")
    print(f"Debug mode: {debug}")
    print("\nAPI Endpoints:")
    print("  GET  /api/health           - Health check")
    print("  GET  /api/catalog          - Get SKU catalog")
    print("  POST /api/match            - Match products")
    print("  POST /api/pricing          - Calculate pricing")
    print("  POST /api/rfp/analyze      - Full RFP analysis")
    print("  POST /api/agents/sales     - Sales agent only")
    print("  POST /api/agents/technical - Technical agent only")
    print("  POST /api/agents/pricing   - Pricing agent only")
    print("=" * 80)
    
    app.run(host=host, port=port, debug=debug)