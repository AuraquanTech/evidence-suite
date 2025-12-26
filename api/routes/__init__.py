"""Evidence Suite - API Routes"""

from .analysis import router as analysis_router
from .cases import router as cases_router
from .evidence import router as evidence_router


__all__ = [
    "analysis_router",
    "cases_router",
    "evidence_router",
]
