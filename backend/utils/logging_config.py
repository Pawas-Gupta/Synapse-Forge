"""Structured logging configuration with JSON formatting and rotation"""
import logging
import os
from logging.handlers import RotatingFileHandler
from pythonjsonlogger import jsonlogger
from datetime import datetime


class CustomJsonFormatter(jsonlogger.JsonFormatter):
    """Custom JSON formatter with additional fields"""
    
    def add_fields(self, log_record, record, message_dict):
        super(CustomJsonFormatter, self).add_fields(log_record, record, message_dict)
        
        # Add timestamp in ISO format
        if not log_record.get('timestamp'):
            log_record['timestamp'] = datetime.utcnow().isoformat() + 'Z'
        
        # Add level name
        if log_record.get('level'):
            log_record['level'] = log_record['level'].upper()
        else:
            log_record['level'] = record.levelname


def setup_logging(log_level=None, log_file='logs/agent_activity.log'):
    """
    Configure application logging with JSON formatting and rotation
    
    Args:
        log_level: Logging level (default: INFO from env or INFO)
        log_file: Path to log file (default: logs/agent_activity.log)
    """
    # Get log level from environment or parameter
    if log_level is None:
        log_level = os.getenv('LOG_LEVEL', 'INFO')
    
    # Convert string level to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create logs directory if it doesn't exist
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Get root logger
    logger = logging.getLogger()
    logger.setLevel(numeric_level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers = []
    
    # Create JSON formatter
    formatter = CustomJsonFormatter(
        '%(timestamp)s %(level)s %(name)s %(message)s'
    )
    
    # File handler with rotation (10MB max, keep 5 backups)
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Log initialization
    logger.info('Logging system initialized', extra={
        'log_level': log_level,
        'log_file': log_file,
        'max_bytes': 10 * 1024 * 1024,
        'backup_count': 5
    })
    
    return logger


def get_logger(name):
    """
    Get a logger instance for a specific module
    
    Args:
        name: Logger name (typically __name__)
    
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


# Convenience function for logging agent execution
def log_agent_execution(logger, agent_name, input_length, output_status, execution_time, **kwargs):
    """
    Log agent execution with standard fields
    
    Args:
        logger: Logger instance
        agent_name: Name of the agent
        input_length: Length of input text
        output_status: Status of output (success/error)
        execution_time: Execution time in seconds
        **kwargs: Additional fields to log
    """
    logger.info(
        f'Agent execution completed: {agent_name}',
        extra={
            'agent_name': agent_name,
            'input_length': input_length,
            'output_status': output_status,
            'execution_time': execution_time,
            **kwargs
        }
    )


# Convenience function for logging LLM API calls
def log_llm_call(logger, model_name, token_count, estimated_cost=None, **kwargs):
    """
    Log LLM API call with standard fields
    
    Args:
        logger: Logger instance
        model_name: Name of the LLM model
        token_count: Number of tokens used
        estimated_cost: Estimated cost in USD (optional)
        **kwargs: Additional fields to log
    """
    extra_fields = {
        'model_name': model_name,
        'token_count': token_count,
        **kwargs
    }
    
    if estimated_cost is not None:
        extra_fields['estimated_cost'] = estimated_cost
    
    logger.info(
        f'LLM API call: {model_name}',
        extra=extra_fields
    )
