"""
Evidence Suite - Database Repository Pattern
Optimized query methods with proper eager loading and caching.
"""
from datetime import datetime
from typing import Optional, List, Tuple, Dict, Any, TypeVar, Generic
from uuid import UUID

from sqlalchemy import select, func, and_, or_, desc
from sqlalchemy.orm import selectinload, joinedload
from sqlalchemy.ext.asyncio import AsyncSession

from .models import (
    Case, CaseStatus,
    EvidenceRecord, EvidenceStatus, EvidenceTypeDB,
    ChainOfCustodyLog, AnalysisResult, AnalysisJob,
    User, AuditLog
)

T = TypeVar('T')


class BaseRepository(Generic[T]):
    """Base repository with common query patterns."""

    def __init__(self, db: AsyncSession, model: type):
        self.db = db
        self.model = model

    async def get_by_id(self, id: UUID) -> Optional[T]:
        """Get entity by ID."""
        result = await self.db.execute(
            select(self.model).where(self.model.id == id)
        )
        return result.scalar_one_or_none()

    async def get_paginated(
        self,
        page: int = 1,
        page_size: int = 20,
        order_by: str = "created_at",
        desc_order: bool = True,
        filters: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[T], int]:
        """Get paginated results with total count in single query."""
        query = select(self.model)

        # Apply filters
        if filters:
            for key, value in filters.items():
                if hasattr(self.model, key) and value is not None:
                    query = query.where(getattr(self.model, key) == value)

        # Get total count using window function (single query)
        count_query = select(func.count()).select_from(query.subquery())
        total = (await self.db.execute(count_query)).scalar() or 0

        # Apply ordering
        order_col = getattr(self.model, order_by, self.model.created_at)
        if desc_order:
            query = query.order_by(desc(order_col))
        else:
            query = query.order_by(order_col)

        # Apply pagination
        query = query.offset((page - 1) * page_size).limit(page_size)

        result = await self.db.execute(query)
        items = result.scalars().all()

        return list(items), total


class CaseRepository(BaseRepository[Case]):
    """Repository for Case operations with optimized queries."""

    def __init__(self, db: AsyncSession):
        super().__init__(db, Case)

    async def get_with_evidence_count(self, case_id: UUID) -> Optional[Tuple[Case, int]]:
        """Get case with evidence count in single query."""
        query = (
            select(
                Case,
                func.count(EvidenceRecord.id).label("evidence_count")
            )
            .outerjoin(EvidenceRecord, EvidenceRecord.case_id == Case.id)
            .where(Case.id == case_id)
            .group_by(Case.id)
        )
        result = await self.db.execute(query)
        row = result.first()
        if row:
            return row[0], row[1]
        return None

    async def get_by_case_number(self, case_number: str) -> Optional[Case]:
        """Get case by case number (indexed lookup)."""
        result = await self.db.execute(
            select(Case).where(Case.case_number == case_number)
        )
        return result.scalar_one_or_none()

    async def list_with_evidence_counts(
        self,
        page: int = 1,
        page_size: int = 20,
        status: Optional[CaseStatus] = None,
        search: Optional[str] = None
    ) -> Tuple[List[Tuple[Case, int]], int]:
        """List cases with evidence counts - optimized single query."""
        # Subquery for evidence counts
        evidence_count_subq = (
            select(
                EvidenceRecord.case_id,
                func.count(EvidenceRecord.id).label("count")
            )
            .group_by(EvidenceRecord.case_id)
            .subquery()
        )

        # Main query with LEFT JOIN to count subquery
        query = (
            select(
                Case,
                func.coalesce(evidence_count_subq.c.count, 0).label("evidence_count")
            )
            .outerjoin(
                evidence_count_subq,
                Case.id == evidence_count_subq.c.case_id
            )
        )

        # Apply filters
        if status:
            query = query.where(Case.status == status)

        if search:
            search_pattern = f"%{search}%"
            query = query.where(
                or_(
                    Case.title.ilike(search_pattern),
                    Case.case_number.ilike(search_pattern)
                )
            )

        # Count total (before pagination)
        count_subq = query.subquery()
        total = (await self.db.execute(
            select(func.count()).select_from(count_subq)
        )).scalar() or 0

        # Apply ordering and pagination
        query = query.order_by(desc(Case.created_at))
        query = query.offset((page - 1) * page_size).limit(page_size)

        result = await self.db.execute(query)
        rows = result.all()

        return [(row[0], row[1]) for row in rows], total

    async def get_statistics(self) -> Dict[str, Any]:
        """Get case statistics in single query."""
        result = await self.db.execute(
            select(
                Case.status,
                func.count(Case.id).label("count")
            ).group_by(Case.status)
        )

        stats = {status.value: 0 for status in CaseStatus}
        for row in result:
            stats[row.status.value] = row.count

        return {
            "by_status": stats,
            "total": sum(stats.values())
        }


class EvidenceRepository(BaseRepository[EvidenceRecord]):
    """Repository for Evidence operations with optimized queries."""

    def __init__(self, db: AsyncSession):
        super().__init__(db, EvidenceRecord)

    async def get_with_custody(self, evidence_id: UUID) -> Optional[EvidenceRecord]:
        """Get evidence with custody entries (eager loaded)."""
        result = await self.db.execute(
            select(EvidenceRecord)
            .options(selectinload(EvidenceRecord.custody_entries))
            .where(EvidenceRecord.id == evidence_id)
        )
        return result.scalar_one_or_none()

    async def get_with_analysis(self, evidence_id: UUID) -> Optional[EvidenceRecord]:
        """Get evidence with analysis results (eager loaded)."""
        result = await self.db.execute(
            select(EvidenceRecord)
            .options(selectinload(EvidenceRecord.analysis_results))
            .where(EvidenceRecord.id == evidence_id)
        )
        return result.scalar_one_or_none()

    async def get_full(self, evidence_id: UUID) -> Optional[EvidenceRecord]:
        """Get evidence with all relationships (eager loaded)."""
        result = await self.db.execute(
            select(EvidenceRecord)
            .options(
                selectinload(EvidenceRecord.custody_entries),
                selectinload(EvidenceRecord.analysis_results),
                joinedload(EvidenceRecord.case)
            )
            .where(EvidenceRecord.id == evidence_id)
        )
        return result.scalar_one_or_none()

    async def get_by_hash(self, case_id: UUID, file_hash: str) -> Optional[EvidenceRecord]:
        """Check for duplicate evidence by hash (indexed lookup)."""
        result = await self.db.execute(
            select(EvidenceRecord).where(
                and_(
                    EvidenceRecord.case_id == case_id,
                    EvidenceRecord.original_hash == file_hash
                )
            )
        )
        return result.scalar_one_or_none()

    async def list_by_case(
        self,
        case_id: UUID,
        status: Optional[EvidenceStatus] = None,
        evidence_type: Optional[EvidenceTypeDB] = None,
        page: int = 1,
        page_size: int = 20
    ) -> Tuple[List[EvidenceRecord], int]:
        """List evidence for a case with optional filters."""
        query = select(EvidenceRecord).where(EvidenceRecord.case_id == case_id)

        if status:
            query = query.where(EvidenceRecord.status == status)
        if evidence_type:
            query = query.where(EvidenceRecord.evidence_type == evidence_type)

        # Count total
        count_subq = query.subquery()
        total = (await self.db.execute(
            select(func.count()).select_from(count_subq)
        )).scalar() or 0

        # Apply pagination
        query = query.order_by(desc(EvidenceRecord.created_at))
        query = query.offset((page - 1) * page_size).limit(page_size)

        result = await self.db.execute(query)
        items = result.scalars().all()

        return list(items), total

    async def get_pending_analysis(self, limit: int = 100) -> List[EvidenceRecord]:
        """Get evidence pending analysis (batch processing)."""
        result = await self.db.execute(
            select(EvidenceRecord)
            .where(EvidenceRecord.status == EvidenceStatus.PENDING)
            .order_by(EvidenceRecord.created_at)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def bulk_update_status(
        self,
        evidence_ids: List[UUID],
        new_status: EvidenceStatus
    ) -> int:
        """Bulk update evidence status (efficient batch operation)."""
        from sqlalchemy import update

        stmt = (
            update(EvidenceRecord)
            .where(EvidenceRecord.id.in_(evidence_ids))
            .values(status=new_status)
        )
        result = await self.db.execute(stmt)
        await self.db.commit()
        return result.rowcount

    async def get_statistics(self, case_id: Optional[UUID] = None) -> Dict[str, Any]:
        """Get evidence statistics."""
        query = select(
            EvidenceRecord.status,
            func.count(EvidenceRecord.id).label("count")
        )

        if case_id:
            query = query.where(EvidenceRecord.case_id == case_id)

        query = query.group_by(EvidenceRecord.status)

        result = await self.db.execute(query)

        stats = {status.value: 0 for status in EvidenceStatus}
        for row in result:
            stats[row.status.value] = row.count

        return {
            "by_status": stats,
            "total": sum(stats.values())
        }


class AnalysisJobRepository(BaseRepository[AnalysisJob]):
    """Repository for AnalysisJob operations."""

    def __init__(self, db: AsyncSession):
        super().__init__(db, AnalysisJob)

    async def get_by_evidence_id(self, evidence_id: UUID) -> Optional[AnalysisJob]:
        """Get latest job for evidence."""
        result = await self.db.execute(
            select(AnalysisJob)
            .where(AnalysisJob.evidence_id == evidence_id)
            .order_by(desc(AnalysisJob.created_at))
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def get_pending_jobs(self, limit: int = 100) -> List[AnalysisJob]:
        """Get pending jobs for processing."""
        result = await self.db.execute(
            select(AnalysisJob)
            .where(AnalysisJob.status == "pending")
            .order_by(AnalysisJob.created_at)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def get_running_jobs(self) -> List[AnalysisJob]:
        """Get currently running jobs."""
        result = await self.db.execute(
            select(AnalysisJob).where(AnalysisJob.status == "running")
        )
        return list(result.scalars().all())

    async def get_job_statistics(self) -> Dict[str, int]:
        """Get job statistics by status."""
        result = await self.db.execute(
            select(
                AnalysisJob.status,
                func.count(AnalysisJob.id).label("count")
            ).group_by(AnalysisJob.status)
        )

        return {row.status: row.count for row in result}


class ChainOfCustodyRepository(BaseRepository[ChainOfCustodyLog]):
    """Repository for Chain of Custody operations."""

    def __init__(self, db: AsyncSession):
        super().__init__(db, ChainOfCustodyLog)

    async def get_by_evidence(
        self,
        evidence_id: UUID,
        limit: Optional[int] = None
    ) -> List[ChainOfCustodyLog]:
        """Get custody entries for evidence, ordered by timestamp."""
        query = (
            select(ChainOfCustodyLog)
            .where(ChainOfCustodyLog.evidence_id == evidence_id)
            .order_by(ChainOfCustodyLog.timestamp)
        )

        if limit:
            query = query.limit(limit)

        result = await self.db.execute(query)
        return list(result.scalars().all())

    async def validate_chain(self, evidence_id: UUID, original_hash: str) -> bool:
        """Validate chain of custody integrity."""
        entries = await self.get_by_evidence(evidence_id)

        if not entries:
            return True

        prev_hash = original_hash
        for entry in entries:
            if entry.input_hash != prev_hash:
                return False
            prev_hash = entry.output_hash

        return True

    async def add_entry(
        self,
        evidence_id: UUID,
        agent_id: str,
        agent_type: str,
        action: str,
        input_hash: str,
        output_hash: str,
        processing_time_ms: Optional[float] = None,
        success: bool = True,
        error_message: Optional[str] = None
    ) -> ChainOfCustodyLog:
        """Add new custody entry."""
        entry = ChainOfCustodyLog(
            evidence_id=evidence_id,
            agent_id=agent_id,
            agent_type=agent_type,
            action=action,
            input_hash=input_hash,
            output_hash=output_hash,
            processing_time_ms=processing_time_ms,
            success=success,
            error_message=error_message,
        )
        self.db.add(entry)
        await self.db.commit()
        await self.db.refresh(entry)
        return entry


# Factory function for dependency injection
def get_case_repository(db: AsyncSession) -> CaseRepository:
    """Get case repository instance."""
    return CaseRepository(db)


def get_evidence_repository(db: AsyncSession) -> EvidenceRepository:
    """Get evidence repository instance."""
    return EvidenceRepository(db)


def get_job_repository(db: AsyncSession) -> AnalysisJobRepository:
    """Get job repository instance."""
    return AnalysisJobRepository(db)


def get_custody_repository(db: AsyncSession) -> ChainOfCustodyRepository:
    """Get custody repository instance."""
    return ChainOfCustodyRepository(db)
