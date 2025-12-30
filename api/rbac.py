"""Evidence Suite - Role-Based Access Control (RBAC)
Middleware and decorators for enforcing role-based permissions.
"""

from collections.abc import Callable
from enum import Enum
from functools import wraps

from fastapi import Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.auth import get_current_user
from core.database import User
from core.database.session import get_db


class Role(str, Enum):
    """User roles with hierarchical permissions."""

    VIEWER = "viewer"  # Read-only access
    ANALYST = "analyst"  # Can analyze evidence
    REVIEWER = "reviewer"  # Can review and verify
    ADMIN = "admin"  # Full access


class Permission(str, Enum):
    """Granular permissions."""

    # Cases
    CASE_READ = "case:read"
    CASE_CREATE = "case:create"
    CASE_UPDATE = "case:update"
    CASE_DELETE = "case:delete"

    # Evidence
    EVIDENCE_READ = "evidence:read"
    EVIDENCE_UPLOAD = "evidence:upload"
    EVIDENCE_ANALYZE = "evidence:analyze"
    EVIDENCE_VERIFY = "evidence:verify"
    EVIDENCE_DELETE = "evidence:delete"

    # Analysis
    ANALYSIS_READ = "analysis:read"
    ANALYSIS_RUN = "analysis:run"
    ANALYSIS_EXPORT = "analysis:export"

    # Admin
    USER_READ = "user:read"
    USER_CREATE = "user:create"
    USER_UPDATE = "user:update"
    USER_DELETE = "user:delete"
    SYSTEM_ADMIN = "system:admin"


# Role -> Permissions mapping
ROLE_PERMISSIONS: dict[Role, set[Permission]] = {
    Role.VIEWER: {
        Permission.CASE_READ,
        Permission.EVIDENCE_READ,
        Permission.ANALYSIS_READ,
    },
    Role.ANALYST: {
        Permission.CASE_READ,
        Permission.CASE_CREATE,
        Permission.CASE_UPDATE,
        Permission.EVIDENCE_READ,
        Permission.EVIDENCE_UPLOAD,
        Permission.EVIDENCE_ANALYZE,
        Permission.ANALYSIS_READ,
        Permission.ANALYSIS_RUN,
        Permission.ANALYSIS_EXPORT,
    },
    Role.REVIEWER: {
        Permission.CASE_READ,
        Permission.CASE_CREATE,
        Permission.CASE_UPDATE,
        Permission.EVIDENCE_READ,
        Permission.EVIDENCE_UPLOAD,
        Permission.EVIDENCE_ANALYZE,
        Permission.EVIDENCE_VERIFY,
        Permission.ANALYSIS_READ,
        Permission.ANALYSIS_RUN,
        Permission.ANALYSIS_EXPORT,
        Permission.USER_READ,
    },
    Role.ADMIN: {
        # Admins have all permissions
        *Permission.__members__.values()
    },
}


def get_role_permissions(role: str) -> set[Permission]:
    """Get permissions for a role."""
    try:
        role_enum = Role(role)
        return ROLE_PERMISSIONS.get(role_enum, set())
    except ValueError:
        return set()


def has_permission(user_role: str, permission: Permission) -> bool:
    """Check if a role has a specific permission."""
    permissions = get_role_permissions(user_role)
    return permission in permissions


class RBACChecker:
    """Dependency for checking permissions."""

    def __init__(self, required_permissions: list[Permission]):
        self.required_permissions = required_permissions

    async def __call__(
        self,
        current_user: dict = Depends(get_current_user),
        db: AsyncSession = Depends(get_db),
    ) -> dict:
        """Check if user has required permissions."""
        user_id = current_user.get("sub")

        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
            )

        # Get user from database
        result = await db.execute(select(User).where(User.email == user_id))
        user = result.scalar_one_or_none()

        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found",
            )

        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User account is disabled",
            )

        # Check permissions
        user_permissions = get_role_permissions(user.role)

        missing_permissions = [p for p in self.required_permissions if p not in user_permissions]

        if missing_permissions:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Missing permissions: {', '.join(p.value for p in missing_permissions)}",
            )

        # Add user info to return
        return {
            **current_user,
            "user_id": str(user.id),
            "role": user.role,
            "permissions": [p.value for p in user_permissions],
        }


def require_permissions(*permissions: Permission):
    """Dependency factory for requiring specific permissions."""
    return RBACChecker(list(permissions))


def require_role(role: Role):
    """Dependency factory for requiring a specific role or higher."""
    role_hierarchy = [Role.VIEWER, Role.ANALYST, Role.REVIEWER, Role.ADMIN]

    try:
        min_index = role_hierarchy.index(role)
        allowed_roles = role_hierarchy[min_index:]
    except ValueError:
        allowed_roles = [role]

    class RoleChecker:
        async def __call__(
            self,
            current_user: dict = Depends(get_current_user),
            db: AsyncSession = Depends(get_db),
        ) -> dict:
            user_id = current_user.get("sub")

            if not user_id:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid authentication credentials",
                )

            result = await db.execute(select(User).where(User.email == user_id))
            user = result.scalar_one_or_none()

            if not user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="User not found",
                )

            if not user.is_active:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="User account is disabled",
                )

            try:
                user_role = Role(user.role)
            except ValueError:
                user_role = Role.VIEWER

            if user_role not in allowed_roles:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Requires role: {role.value} or higher",
                )

            return {
                **current_user,
                "user_id": str(user.id),
                "role": user.role,
            }

    return RoleChecker()


# Convenience dependencies
require_viewer = require_role(Role.VIEWER)
require_analyst = require_role(Role.ANALYST)
require_reviewer = require_role(Role.REVIEWER)
require_admin = require_role(Role.ADMIN)


# Permission-specific dependencies
can_read_cases = require_permissions(Permission.CASE_READ)
can_manage_cases = require_permissions(Permission.CASE_CREATE, Permission.CASE_UPDATE)
can_upload_evidence = require_permissions(Permission.EVIDENCE_UPLOAD)
can_analyze_evidence = require_permissions(Permission.EVIDENCE_ANALYZE)
can_verify_evidence = require_permissions(Permission.EVIDENCE_VERIFY)
can_export_analysis = require_permissions(Permission.ANALYSIS_EXPORT)
can_manage_users = require_permissions(Permission.USER_CREATE, Permission.USER_UPDATE)
