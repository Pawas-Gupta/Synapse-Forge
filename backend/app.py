"""Flask REST API Backend for RFP Agent System"""
import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from prometheus_flask_exporter import PrometheusMetrics
from dotenv import load_dotenv
import json

from backend.database import get_db
from backend.agents.main_agent import MainAgent
from backend.agents.sales_agent import SalesAgent
from backend.agents.technical_agent import TechnicalAgent
from backend.agents.pricing_agent import PricingAgent
from backend.tools.sku_tools import match_sku, estimate_cost, get_catalog_summary
from backend.auth import auth, verify_token

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize Prometheus metrics
metrics = PrometheusMetrics(app)

# Add custom metrics
metrics.info('rfp_agent_system_info', 'RFP Agent System Information', version='1.0.0')

# Custom metrics for agents
agent_duration = metrics.histogram(
    'agent_execution_duration_seconds',
    'Agent execution duration in seconds',
    labels={'agent_type': lambda: 'unknown'}
)

agent_errors = metrics.counter(
    'agent_errors_total',
    'Total number of agent errors',
    labels={'agent_type': lambda: 'unknown'}
)

# Initialize rate limiter
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["100 per hour"],  # Global rate limit
    storage_uri=os.getenv('REDIS_URL', 'memory://'),  # Use Redis if available, otherwise in-memory
    strategy="fixed-window"
)

# Global variables for agents
db = None
agents = {}


# Error handlers
@app.errorhandler(400)
def bad_request_handler(e):
    """Handle bad request errors"""
    return jsonify({
        'status': 'error',
        'error': 'Bad Request',
        'message': str(e.description) if hasattr(e, 'description') else 'Invalid request'
    }), 400


@app.errorhandler(401)
def unauthorized_handler(e):
    """Handle unauthorized errors"""
    return jsonify({
        'status': 'error',
        'error': 'Unauthorized',
        'message': 'Authentication required'
    }), 401


@app.errorhandler(403)
def forbidden_handler(e):
    """Handle forbidden errors"""
    return jsonify({
        'status': 'error',
        'error': 'Forbidden',
        'message': 'Insufficient permissions'
    }), 403


@app.errorhandler(404)
def not_found_handler(e):
    """Handle not found errors"""
    return jsonify({
        'status': 'error',
        'error': 'Not Found',
        'message': 'The requested resource was not found'
    }), 404


@app.errorhandler(429)
def ratelimit_handler(e):
    """Handle rate limit exceeded errors"""
    return jsonify({
        'status': 'error',
        'error': 'Rate limit exceeded',
        'message': str(e.description),
        'retry_after': e.description.split('in ')[-1] if 'in ' in str(e.description) else 'unknown'
    }), 429


@app.errorhandler(500)
def internal_error_handler(e):
    """Handle internal server errors"""
    return jsonify({
        'status': 'error',
        'error': 'Internal Server Error',
        'message': 'An unexpected error occurred'
    }), 500


def is_admin_request():
    """Check if request has admin privileges (bypasses rate limiting)"""
    admin_token = os.getenv('ADMIN_API_TOKEN')
    if not admin_token:
        return False
    
    auth_header = request.headers.get('Authorization', '')
    if auth_header.startswith('Bearer '):
        token = auth_header[7:]
        return token == admin_token
    return False


# Custom rate limit exemption for admin
@limiter.request_filter
def admin_whitelist():
    """Exempt admin requests from rate limiting"""
    return is_admin_request()


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
@limiter.exempt  # Health check should not be rate limited
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'RFP Agent System API',
        'version': '1.0.0'
    })


@app.route('/api/catalog', methods=['GET'])
@limiter.limit("50 per minute")  # Catalog endpoint: 50 requests per minute
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
@limiter.limit("10 per minute")  # Analyze endpoint: 10 requests per minute
@auth.login_required  # Require authentication
async def analyze_rfp():
    """Analyze complete RFP using all agents (async)"""
    try:
        data = request.get_json()
        rfp_input = data.get('rfp_input', '')
        session_id = data.get('session_id', None)
        
        if not rfp_input:
            return jsonify({
                'status': 'error',
                'error': 'rfp_input is required'
            }), 400
        
        initialize_agents()
        
        # Set session ID if provided
        if session_id:
            agents['main'].set_session_id(session_id)
        
        # Orchestrate workflow (async)
        result = await agents['main'].orchestrate(rfp_input)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500


@app.route('/api/agents/sales', methods=['POST'])
@auth.login_required
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
@auth.login_required
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
@auth.login_required
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


@app.route('/api/conversation/history', methods=['GET'])
def get_conversation_history():
    """Get conversation history for current session"""
    try:
        session_id = request.args.get('session_id')
        
        if not session_id:
            return jsonify({
                'status': 'error',
                'error': 'session_id is required'
            }), 400
        
        initialize_agents()
        
        # Set session ID
        agents['main'].set_session_id(session_id)
        
        # Get history
        history = agents['main'].get_conversation_history()
        
        return jsonify({
            'status': 'success',
            'session_id': session_id,
            'message_count': len(history),
            'history': history
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500


@app.route('/api/conversation/clear', methods=['POST'])
def clear_conversation():
    """Clear conversation history for a session"""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        
        if not session_id:
            return jsonify({
                'status': 'error',
                'error': 'session_id is required'
            }), 400
        
        initialize_agents()
        
        # Set session ID and clear
        agents['main'].set_session_id(session_id)
        agents['main'].clear_memory()
        
        return jsonify({
            'status': 'success',
            'message': f'Conversation history cleared for session {session_id}'
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500


@app.route('/api/cache-metrics', methods=['GET'])
@limiter.exempt
def get_cache_metrics():
    """Get cache hit/miss metrics"""
    try:
        from backend.utils.cache import get_cache_manager
        cache_manager = get_cache_manager()
        metrics = cache_manager.get_metrics()
        
        return jsonify({
            'status': 'success',
            'metrics': metrics
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500


@app.route('/api/pool-status', methods=['GET'])
@limiter.exempt
def get_pool_status():
    """Get database connection pool status"""
    try:
        from backend.database import get_db_pool
        pool = get_db_pool()
        status = pool.get_pool_status()
        
        return jsonify({
            'status': 'success',
            'pool_status': status
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500


# ============================================================================
# RUN SERVER
# ============================================================================

def shutdown_handler(signum, frame):
    """Handle graceful shutdown"""
    print("\n" + "=" * 80)
    print("Shutting down gracefully...")
    print("=" * 80)
    
    # Close database connections
    global db
    if db:
        try:
            db.close()
            print("✓ Database connections closed")
        except Exception as e:
            print(f"⚠ Error closing database: {e}")
    
    # Close cache connections
    try:
        from backend.utils.cache import get_cache_manager
        cache = get_cache_manager()
        if hasattr(cache, 'redis_client') and cache.redis_client:
            cache.redis_client.close()
            print("✓ Cache connections closed")
    except Exception as e:
        print(f"⚠ Error closing cache: {e}")
    
    print("Shutdown complete")
    sys.exit(0)


if __name__ == '__main__':
    import signal
    
    # Register shutdown handlers
    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)
    
    host = os.getenv('FLASK_HOST', '127.0.0.1')
    port = int(os.getenv('FLASK_PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'
    
    print("=" * 80)
    print("RFP Agent System - Flask Backend")
    print("=" * 80)
    print(f"Server running on: http://{host}:{port}")
    print(f"Debug mode: {debug}")
    print("\nProduction Features:")
    print("  ✓ Rate Limiting (100/hour global, endpoint-specific)")
    print("  ✓ Authentication (Bearer token)")
    print("  ✓ Prometheus Metrics (/metrics)")
    print("  ✓ Async Execution")
    print("  ✓ Conversation Memory")
    print("  ✓ Database Connection Pooling")
    print("  ✓ Caching (Redis/Memory)")
    print("  ✓ Retry & Circuit Breaker")
    print("\nAPI Endpoints:")
    print("  GET  /api/health              - Health check")
    print("  GET  /metrics                 - Prometheus metrics")
    print("  GET  /api/catalog             - Get SKU catalog")
    print("  GET  /api/cache-metrics       - Cache statistics")
    print("  GET  /api/pool-status         - DB pool status")
    print("  POST /api/rfp/analyze         - Full RFP analysis (auth required)")
    print("  POST /api/agents/*            - Individual agents (auth required)")
    print("  GET  /api/conversation/*      - Conversation management")
    print("=" * 80)
    print("\nPress Ctrl+C to shutdown gracefully")
    print("=" * 80)
    
    try:
        app.run(host=host, port=port, debug=debug)
    except KeyboardInterrupt:
        shutdown_handler(None, None)