"""
Evidence Suite - API Routes
"""
from .cases import router as cases_router
from .evidence import router as evidence_router
from .analysis import router as analysis_router

__all__ = [
    "cases_router",
    "evidence_router",
    "analysis_router",
]
