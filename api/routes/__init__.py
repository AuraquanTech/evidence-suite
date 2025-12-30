"""Evidence Suite - API Routes"""

from .analysis import router as analysis_router
from .cases import router as cases_router
from .evidence import router as evidence_router
from .exports import router as exports_router
from .reports import router as reports_router


__all__ = [
    "analysis_router",
    "cases_router",
    "evidence_router",
    "exports_router",
    "reports_router",
]
