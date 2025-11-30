"""Authentication module for API endpoints"""
import os
import hashlib
import time
from datetime import datetime, timedelta
from flask_httpauth import HTTPTokenAuth
from backend.utils.logging_config import get_logger

logger = get_logger(__name__)

# Initialize token auth
auth = HTTPTokenAuth(scheme='Bearer')

# Token storage (in production, use database)
# Format: {token_hash: {'user_id': str, 'role': str, 'expires_at': timestamp}}
_tokens = {}


def hash_token(token: str) -> str:
    """Hash a token using SHA256"""
    return hashlib.sha256(token.encode()).hexdigest()


def load_tokens_from_env():
    """Load API tokens from environment variables"""
    global _tokens
    
    # Load admin token
    admin_token = os.getenv('ADMIN_API_TOKEN')
    if admin_token:
        token_hash = hash_token(admin_token)
        _tokens[token_hash] = {
            'user_id': 'admin',
            'role': 'admin',
            'expires_at': None  # Admin tokens don't expire
        }
        logger.info('Admin API token loaded')
    
    # Load user tokens (comma-separated)
    user_tokens = os.getenv('USER_API_TOKENS', '')
    if user_tokens:
        for i, token in enumerate(user_tokens.split(',')):
            token = token.strip()
            if token:
                token_hash = hash_token(token)
                _tokens[token_hash] = {
                    'user_id': f'user_{i+1}',
                    'role': 'user',
                    'expires_at': time.time() + (24 * 3600)  # 24 hours
                }
        logger.info(f'Loaded {len(user_tokens.split(","))} user API tokens')


def create_token(user_id: str, role: str = 'user', expires_hours: int = 24) -> str:
    """
    Create a new API token
    
    Args:
        user_id: User identifier
        role: User role ('user' or 'admin')
        expires_hours: Token expiration in hours
    
    Returns:
        Generated token string
    """
    import secrets
    token = secrets.token_urlsafe(32)
    token_hash = hash_token(token)
    
    expires_at = time.time() + (expires_hours * 3600) if expires_hours > 0 else None
    
    _tokens[token_hash] = {
        'user_id': user_id,
        'role': role,
        'expires_at': expires_at
    }
    
    logger.info(f'Created token for user {user_id} with role {role}')
    return token


def revoke_token(token: str):
    """Revoke an API token"""
    token_hash = hash_token(token)
    if token_hash in _tokens:
        user_id = _tokens[token_hash]['user_id']
        del _tokens[token_hash]
        logger.info(f'Revoked token for user {user_id}')
        return True
    return False


@auth.verify_token
def verify_token(token):
    """
    Verify Bearer token
    
    Returns user info if valid, None otherwise
    """
    if not token:
        return None
    
    token_hash = hash_token(token)
    
    if token_hash not in _tokens:
        logger.warning(f'Invalid token attempt')
        return None
    
    token_info = _tokens[token_hash]
    
    # Check expiration
    if token_info['expires_at'] is not None:
        if time.time() > token_info['expires_at']:
            logger.warning(f'Expired token for user {token_info["user_id"]}')
            del _tokens[token_hash]
            return None
    
    logger.info(f'Authenticated user {token_info["user_id"]} with role {token_info["role"]}')
    return token_info


@auth.get_user_roles
def get_user_roles(user_info):
    """Get user roles for authorization"""
    if user_info:
        return [user_info['role']]
    return []


def require_role(role: str):
    """
    Decorator to require specific role
    
    Args:
        role: Required role ('user' or 'admin')
    """
    def decorator(f):
        from functools import wraps
        @wraps(f)
        def decorated_function(*args, **kwargs):
            user_info = auth.current_user()
            if not user_info or user_info['role'] != role:
                from flask import jsonify
                logger.warning(f'Insufficient permissions for user {user_info.get("user_id") if user_info else "unknown"}')
                return jsonify({
                    'status': 'error',
                    'error': 'Insufficient permissions',
                    'message': f'This endpoint requires {role} role'
                }), 403
            return f(*args, **kwargs)
        return decorated_function
    return decorator


# Load tokens on module import
load_tokens_from_env()
