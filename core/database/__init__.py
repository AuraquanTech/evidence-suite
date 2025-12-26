"""
Evidence Suite - Database Module
With optimized repositories and connection monitoring.
"""
from .models import (
    Base,
    Case,
    CaseStatus,
    EvidenceRecord,
    EvidenceStatus,
    EvidenceTypeDB,
    ChainOfCustodyLog,
    AnalysisResult,
    AnalysisJob,
    User,
    AuditLog,
)
from .session import (
    get_engine,
    get_session,
    get_async_session_local,
    init_db,
    init_db_async,
    get_db,
)
from .repository import (
    CaseRepository,
    EvidenceRepository,
    AnalysisJobRepository,
    ChainOfCustodyRepository,
    get_case_repository,
    get_evidence_repository,
    get_job_repository,
    get_custody_repository,
)
from .monitoring import (
    DatabaseMonitor,
    get_monitor,
    setup_pool_monitoring,
    start_monitoring_task,
)

__all__ = [
    # Models
    "Base",
    "Case",
    "CaseStatus",
    "EvidenceRecord",
    "EvidenceStatus",
    "EvidenceTypeDB",
    "ChainOfCustodyLog",
    "AnalysisResult",
    "AnalysisJob",
    "User",
    "AuditLog",
    # Session
    "get_engine",
    "get_session",
    "get_async_session_local",
    "init_db",
    "init_db_async",
    "get_db",
    # Repositories
    "CaseRepository",
    "EvidenceRepository",
    "AnalysisJobRepository",
    "ChainOfCustodyRepository",
    "get_case_repository",
    "get_evidence_repository",
    "get_job_repository",
    "get_custody_repository",
    # Monitoring
    "DatabaseMonitor",
    "get_monitor",
    "setup_pool_monitoring",
    "start_monitoring_task",
]
