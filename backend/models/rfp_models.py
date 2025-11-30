"""Pydantic models for RFP data structures"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class RFPRequirement(BaseModel):
    """Individual requirement from an RFP"""
    item: str = Field(..., description="Item or component name")
    quantity: Optional[int] = Field(None, description="Quantity required")
    specifications: Dict[str, Any] = Field(default_factory=dict, description="Technical specifications")
    description: Optional[str] = Field(None, description="Detailed description")


class RFPParsed(BaseModel):
    """Structured RFP document"""
    title: str = Field(..., description="RFP title or project name")
    rfp_number: Optional[str] = Field(None, description="RFP identification number")
    budget_min: Optional[float] = Field(None, description="Minimum budget in USD")
    budget_max: Optional[float] = Field(None, description="Maximum budget in USD")
    timeline: Optional[str] = Field(None, description="Project timeline or deadline")
    requirements: List[RFPRequirement] = Field(default_factory=list, description="List of requirements")
    compliance: List[str] = Field(default_factory=list, description="Compliance requirements")
    contact: Optional[str] = Field(None, description="Contact information")
    raw_text: Optional[str] = Field(None, description="Original RFP text")
    
    class Config:
        json_schema_extra = {
            "example": {
                "title": "Manufacturing Facility Electrical System Upgrade",
                "rfp_number": "RFP-2025-001",
                "budget_min": 60000.0,
                "budget_max": 80000.0,
                "timeline": "6-8 weeks from award",
                "requirements": [
                    {
                        "item": "Circuit Breaker",
                        "quantity": 10,
                        "specifications": {"voltage": "480V", "current": "100A"}
                    }
                ],
                "compliance": ["NEC 2020", "UL listed"],
                "contact": "procurement@example.com"
            }
        }


class RFPDocument(BaseModel):
    """Represents a discovered RFP document"""
    url: str = Field(..., description="Source URL")
    title: str = Field(..., description="Document title")
    content: str = Field(..., description="Document content")
    format: str = Field(..., description="Document format (html, pdf, doc, mock)")
    discovered_at: datetime = Field(default_factory=datetime.utcnow, description="Discovery timestamp")
