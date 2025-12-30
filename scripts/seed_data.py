#!/usr/bin/env python3
"""Evidence Suite - Seed Data Script
Populate database with sample data for development/testing.
"""

import asyncio
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4


# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from passlib.context import CryptContext


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# Sample data
SAMPLE_USERS = [
    {
        "email": "admin@evidencesuite.local",
        "name": "System Administrator",
        "role": "admin",
        "password": "admin123",
    },
    {
        "email": "analyst@evidencesuite.local",
        "name": "Jane Analyst",
        "role": "analyst",
        "password": "analyst123",
    },
    {
        "email": "reviewer@evidencesuite.local",
        "name": "John Reviewer",
        "role": "reviewer",
        "password": "reviewer123",
    },
    {
        "email": "viewer@evidencesuite.local",
        "name": "View Only User",
        "role": "viewer",
        "password": "viewer123",
    },
]

SAMPLE_CASES = [
    {
        "case_number": "CASE-2024-001",
        "title": "Smith vs. Johnson - Custody Dispute",
        "description": "Analysis of communication patterns in custody dispute case.",
        "client_name": "Smith Family Trust",
        "attorney_name": "Sarah Mitchell, Esq.",
        "jurisdiction": "California",
        "status": "active",
    },
    {
        "case_number": "CASE-2024-002",
        "title": "Workplace Harassment Investigation",
        "description": "Review of email and message evidence for HR investigation.",
        "client_name": "Acme Corporation",
        "attorney_name": "Robert Chen, Esq.",
        "jurisdiction": "New York",
        "status": "open",
    },
    {
        "case_number": "CASE-2024-003",
        "title": "Insurance Fraud Review",
        "description": "Analysis of claimant communications for fraud indicators.",
        "client_name": "SecureLife Insurance",
        "attorney_name": "Michelle Davis, Esq.",
        "jurisdiction": "Texas",
        "status": "pending_review",
    },
]

SAMPLE_EVIDENCE_TEXTS = [
    {
        "filename": "email_thread_1.txt",
        "content": """
From: john@example.com
To: jane@example.com
Subject: Re: Our conversation

I never said that. You're always making things up. You're too sensitive.
Maybe you should see someone about your memory issues.

I'm the one who's been hurt here. After everything I've done for you,
this is how you treat me? You made me do this.
        """,
        "type": "email",
    },
    {
        "filename": "message_log.txt",
        "content": """
[10:32 AM] User1: Hey, can we talk about what happened?
[10:35 AM] User2: That never happened. You're imagining things.
[10:36 AM] User1: I have the messages saved...
[10:38 AM] User2: Those are fake. You're crazy. Everyone thinks so.
[10:40 AM] User2: If you loved me, you wouldn't bring this up.
[10:42 AM] User2: Look what you did. Now I'm upset. This is your fault.
        """,
        "type": "text",
    },
    {
        "filename": "statement_analysis.txt",
        "content": """
I want to be completely honest with you. Believe me when I say I had nothing
to do with this. Truthfully, I was nowhere near there. That person must have
been someone else.

I did not take the money. I would not ever do such a thing. One might say
it was someone from the other department. They say it happens all the time.
        """,
        "type": "document",
    },
]


async def seed_database(clear_existing: bool = False):
    """Seed the database with sample data."""
    from sqlalchemy import delete, select

    from core.database import (
        AnalysisJob,
        AnalysisResult,
        AuditLog,
        Case,
        CaseStatus,
        ChainOfCustodyLog,
        EvidenceRecord,
        EvidenceStatus,
        EvidenceTypeDB,
        User,
    )
    from core.database.session import get_async_session, init_db_async

    print("Initializing database...")
    await init_db_async()

    async with get_async_session() as db:
        if clear_existing:
            print("Clearing existing data...")
            # Delete in order of dependencies
            await db.execute(delete(AuditLog))
            await db.execute(delete(AnalysisResult))
            await db.execute(delete(AnalysisJob))
            await db.execute(delete(ChainOfCustodyLog))
            await db.execute(delete(EvidenceRecord))
            await db.execute(delete(Case))
            await db.execute(delete(User))
            await db.commit()
            print("Existing data cleared.")

        # Create users
        print("\nCreating users...")
        users = []
        for user_data in SAMPLE_USERS:
            # Check if user exists
            existing = await db.execute(select(User).where(User.email == user_data["email"]))
            if existing.scalar_one_or_none():
                print(f"  User {user_data['email']} already exists, skipping...")
                continue

            user = User(
                id=uuid4(),
                email=user_data["email"],
                name=user_data["name"],
                role=user_data["role"],
                password_hash=pwd_context.hash(user_data["password"]),
                is_active=True,
                created_at=datetime.utcnow(),
            )
            db.add(user)
            users.append(user)
            print(f"  Created user: {user_data['email']} ({user_data['role']})")

        await db.commit()

        # Create cases
        print("\nCreating cases...")
        cases = []
        for i, case_data in enumerate(SAMPLE_CASES):
            # Check if case exists
            existing = await db.execute(
                select(Case).where(Case.case_number == case_data["case_number"])
            )
            if existing.scalar_one_or_none():
                print(f"  Case {case_data['case_number']} already exists, skipping...")
                continue

            case = Case(
                id=uuid4(),
                case_number=case_data["case_number"],
                title=case_data["title"],
                description=case_data["description"],
                client_name=case_data["client_name"],
                attorney_name=case_data["attorney_name"],
                jurisdiction=case_data["jurisdiction"],
                status=CaseStatus(case_data["status"]),
                created_at=datetime.utcnow() - timedelta(days=30 - i * 10),
            )
            db.add(case)
            cases.append(case)
            print(f"  Created case: {case_data['case_number']}")

        await db.commit()

        # Create evidence for first case
        if cases:
            print("\nCreating evidence records...")

            # Ensure evidence store directory exists
            evidence_dir = Path("./evidence_store") / str(cases[0].id)
            evidence_dir.mkdir(parents=True, exist_ok=True)

            for i, evidence_data in enumerate(SAMPLE_EVIDENCE_TEXTS):
                # Save content to file
                file_path = evidence_dir / evidence_data["filename"]
                content = evidence_data["content"].strip().encode()
                file_path.write_bytes(content)

                # Calculate hash
                import hashlib

                file_hash = hashlib.sha256(content).hexdigest()

                # Create evidence record
                evidence = EvidenceRecord(
                    id=uuid4(),
                    case_id=cases[0].id,
                    evidence_type=EvidenceTypeDB(evidence_data["type"]),
                    original_filename=evidence_data["filename"],
                    mime_type="text/plain",
                    file_size_bytes=len(content),
                    original_hash=file_hash,
                    storage_path=str(file_path),
                    status=EvidenceStatus.PENDING,
                    extracted_text=evidence_data["content"].strip(),
                    created_at=datetime.utcnow() - timedelta(days=25 - i * 5),
                )
                db.add(evidence)

                # Create chain of custody entry
                custody = ChainOfCustodyLog(
                    evidence_id=evidence.id,
                    timestamp=datetime.utcnow(),
                    agent_id="seed-script",
                    agent_type="system",
                    action="evidence_created",
                    input_hash=file_hash,
                    output_hash=file_hash,
                    success=True,
                )
                db.add(custody)

                print(f"  Created evidence: {evidence_data['filename']}")

            await db.commit()

        print("\n" + "=" * 50)
        print("SEED DATA COMPLETE")
        print("=" * 50)
        print("\nSample Login Credentials:")
        print("-" * 50)
        for user_data in SAMPLE_USERS:
            print(f"  {user_data['role']:10} - {user_data['email']}")
            print(f"             Password: {user_data['password']}")
        print("-" * 50)


async def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Seed Evidence Suite database")
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing data before seeding",
    )
    args = parser.parse_args()

    # Set test environment if not already set
    if not os.getenv("EVIDENCE_SUITE_ENV"):
        os.environ["EVIDENCE_SUITE_ENV"] = "development"

    await seed_database(clear_existing=args.clear)


if __name__ == "__main__":
    asyncio.run(main())
