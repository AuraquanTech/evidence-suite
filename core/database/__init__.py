"""Evidence Suite - Database Module
With optimized repositories and connection monitoring.
"""

from .models import (
    AnalysisJob,
    AnalysisResult,
    AuditLog,
    Base,
    Case,
    CaseStatus,
    ChainOfCustodyLog,
    EvidenceRecord,
    EvidenceStatus,
    EvidenceTypeDB,
    User,
)
from .monitoring import (
    DatabaseMonitor,
    get_monitor,
    setup_pool_monitoring,
    start_monitoring_task,
)
from .repository import (
    AnalysisJobRepository,
    CaseRepository,
    ChainOfCustodyRepository,
    EvidenceRepository,
    get_case_repository,
    get_custody_repository,
    get_evidence_repository,
    get_job_repository,
)
from .session import (
    get_async_session_local,
    get_db,
    get_engine,
    get_session,
    init_db,
    init_db_async,
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
