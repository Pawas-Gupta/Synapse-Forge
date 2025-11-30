"""Validation utilities and Pydantic schemas for agent outputs"""
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional
from datetime import datetime


class SalesAgentOutput(BaseModel):
    """Validation schema for Sales Agent output"""
    status: str
    agent: str
    output: str
    parsed_data: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('status')
    def status_must_be_valid(cls, v):
        if v not in ['success', 'error']:
            raise ValueError(f'Invalid status: {v}. Must be "success" or "error"')
        return v
    
    @validator('agent')
    def agent_must_be_sales(cls, v):
        if v != 'sales':
            raise ValueError(f'Invalid agent: {v}. Must be "sales"')
        return v
    
    @validator('output')
    def output_not_empty(cls, v, values):
        # Only validate output length if status is success
        if values.get('status') == 'success' and (not v or len(v) < 10):
            raise ValueError('Output too short for successful execution')
        return v


class TechnicalAgentOutput(BaseModel):
    """Validation schema for Technical Agent output"""
    status: str
    agent: str
    output: str = ""
    matches: List[Dict[str, Any]] = Field(default_factory=list)
    avg_match_score: float = 0.0
    total_matched: int = 0
    
    @validator('status')
    def status_must_be_valid(cls, v):
        if v not in ['success', 'error']:
            raise ValueError(f'Invalid status: {v}. Must be "success" or "error"')
        return v
    
    @validator('agent')
    def agent_must_be_technical(cls, v):
        if v != 'technical':
            raise ValueError(f'Invalid agent: {v}. Must be "technical"')
        return v
    
    @validator('avg_match_score')
    def score_in_range(cls, v, values):
        # Only validate score if status is success
        if values.get('status') == 'success' and not (0 <= v <= 100):
            raise ValueError(f'Score out of range: {v}. Must be between 0 and 100')
        return v
    
    @validator('matches')
    def matches_not_empty_on_success(cls, v, values):
        # Only validate matches if status is success
        # Allow empty matches if avg_match_score is 0 (no matches above threshold)
        if values.get('status') == 'success' and len(v) == 0:
            if values.get('avg_match_score', 0) > 0:
                raise ValueError('Matches list cannot be empty when avg_match_score > 0')
        return v


class PricingAgentOutput(BaseModel):
    """Validation schema for Pricing Agent output"""
    status: str
    agent: str
    output: str = ""
    total_cost: float = 0.0
    margin: float = 0.0
    final_price: float = 0.0
    breakdown: List[Dict[str, Any]] = Field(default_factory=list)
    
    @validator('status')
    def status_must_be_valid(cls, v):
        if v not in ['success', 'error']:
            raise ValueError(f'Invalid status: {v}. Must be "success" or "error"')
        return v
    
    @validator('agent')
    def agent_must_be_pricing(cls, v):
        if v != 'pricing':
            raise ValueError(f'Invalid agent: {v}. Must be "pricing"')
        return v
    
    @validator('total_cost', 'margin', 'final_price')
    def cost_non_negative(cls, v, values):
        # Only validate costs if status is success
        if values.get('status') == 'success' and v < 0:
            raise ValueError(f'Cost values cannot be negative: {v}')
        return v
    
    @validator('final_price')
    def final_price_consistent(cls, v, values):
        # Only validate consistency if status is success
        if values.get('status') == 'success':
            total_cost = values.get('total_cost', 0)
            margin = values.get('margin', 0)
            expected = total_cost + margin
            # Allow small floating point differences
            if abs(v - expected) > 0.01:
                raise ValueError(
                    f'Final price {v} inconsistent with total_cost {total_cost} + margin {margin}'
                )
        return v


class ValidationError(Exception):
    """Custom exception for validation errors"""
    def __init__(self, message, stage=None, field=None, details=None):
        self.message = message
        self.stage = stage
        self.field = field
        self.details = details
        super().__init__(self.message)
    
    def to_dict(self):
        """Convert to dictionary for error response"""
        return {
            'status': 'error',
            'error_type': 'ValidationError',
            'message': self.message,
            'details': {
                'failed_stage': self.stage,
                'field': self.field,
                'reason': self.details
            },
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        }


def validate_agent_output(output: Dict[str, Any], agent_type: str) -> tuple[bool, Optional[str]]:
    """
    Validate agent output against schema
    
    Args:
        output: Agent output dictionary
        agent_type: Type of agent ('sales', 'technical', 'pricing')
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    schema_map = {
        'sales': SalesAgentOutput,
        'technical': TechnicalAgentOutput,
        'pricing': PricingAgentOutput
    }
    
    if agent_type not in schema_map:
        return False, f'Unknown agent type: {agent_type}'
    
    schema = schema_map[agent_type]
    
    try:
        # Validate against schema
        schema(**output)
        return True, None
    except Exception as e:
        return False, str(e)


def validate_pipeline_stage(stage_name: str, output: Dict[str, Any]) -> None:
    """
    Validate pipeline stage output and raise ValidationError if invalid
    
    Args:
        stage_name: Name of the pipeline stage
        output: Stage output dictionary
    
    Raises:
        ValidationError: If validation fails
    """
    # Determine agent type from stage name
    agent_type = stage_name.lower()
    
    is_valid, error_message = validate_agent_output(output, agent_type)
    
    if not is_valid:
        raise ValidationError(
            message=f'Validation failed for {stage_name} agent',
            stage=stage_name,
            field='output',
            details=error_message
        )
    
    # Check for critical error status
    if output.get('status') == 'error':
        raise ValidationError(
            message=f'{stage_name} agent returned error status',
            stage=stage_name,
            field='status',
            details=output.get('error', 'Unknown error')
        )


def create_error_response(error: Exception, stage: str = None) -> Dict[str, Any]:
    """
    Create structured error response
    
    Args:
        error: Exception that occurred
        stage: Pipeline stage where error occurred
    
    Returns:
        Structured error response dictionary
    """
    if isinstance(error, ValidationError):
        return error.to_dict()
    
    return {
        'status': 'error',
        'error_type': type(error).__name__,
        'message': str(error),
        'details': {
            'failed_stage': stage
        },
        'timestamp': datetime.utcnow().isoformat() + 'Z'
    }
