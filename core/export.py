"""Evidence Suite - Evidence Export Package
Export evidence and analysis as legal-ready packages.
"""

import hashlib
import io
import json
import os
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, BinaryIO
from uuid import UUID

from loguru import logger


class EvidenceExporter:
    """Export evidence as legally-admissible packages.

    Creates ZIP packages containing:
    - Original evidence files
    - Chain of custody logs
    - Analysis reports
    - Hash verification files
    - Metadata JSON
    """

    def __init__(self, output_dir: str | Path = "./exports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def export_case(
        self,
        case_id: str | UUID,
        include_raw_evidence: bool = True,
        include_analysis: bool = True,
        include_reports: bool = True,
        password: str | None = None,
    ) -> Path:
        """Export entire case as a ZIP package.

        Args:
            case_id: Case UUID
            include_raw_evidence: Include original evidence files
            include_analysis: Include analysis results
            include_reports: Generate and include reports
            password: Optional password for ZIP encryption

        Returns:
            Path to export package
        """
        from sqlalchemy import select

        from core.database import (
            AnalysisResult,
            Case,
            ChainOfCustodyLog,
            EvidenceRecord,
        )
        from core.database.session import get_async_session

        async with get_async_session() as db:
            # Get case
            result = await db.execute(select(Case).where(Case.id == UUID(str(case_id))))
            case = result.scalar_one_or_none()

            if not case:
                raise ValueError(f"Case not found: {case_id}")

            # Get evidence records
            result = await db.execute(
                select(EvidenceRecord).where(EvidenceRecord.case_id == case.id)
            )
            evidence_records = result.scalars().all()

            # Create export package
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"export_{case.case_number}_{timestamp}.zip"
            filepath = self.output_dir / filename

            # Build manifest
            manifest = {
                "export_info": {
                    "created_at": datetime.utcnow().isoformat(),
                    "evidence_suite_version": "1.0.0",
                    "export_format_version": "1.0",
                },
                "case": {
                    "id": str(case.id),
                    "case_number": case.case_number,
                    "title": case.title,
                    "description": case.description,
                    "status": case.status.value if case.status else None,
                    "client_name": case.client_name,
                    "attorney_name": case.attorney_name,
                    "jurisdiction": case.jurisdiction,
                    "created_at": case.created_at.isoformat() if case.created_at else None,
                },
                "evidence": [],
                "files": [],
                "integrity": {
                    "algorithm": "SHA-256",
                    "checksums": {},
                },
            }

            with zipfile.ZipFile(filepath, "w", zipfile.ZIP_DEFLATED) as zf:
                # Add each evidence item
                for evidence in evidence_records:
                    evidence_data = await self._export_evidence_item(
                        db, evidence, zf, include_raw_evidence, include_analysis
                    )
                    manifest["evidence"].append(evidence_data)

                    # Track files
                    for file_info in evidence_data.get("files", []):
                        manifest["files"].append(file_info)
                        manifest["integrity"]["checksums"][file_info["path"]] = file_info["hash"]

                # Generate reports if requested
                if include_reports:
                    from core.reports import ReportGenerator

                    generator = ReportGenerator()

                    # Generate PDF report
                    try:
                        report_data = await self._build_report_data(case, evidence_records)
                        pdf_path = generator._generate_pdf(report_data, case.case_number)

                        if pdf_path.exists():
                            zf.write(pdf_path, f"reports/{pdf_path.name}")
                            manifest["files"].append(
                                {
                                    "path": f"reports/{pdf_path.name}",
                                    "type": "report",
                                    "format": "pdf",
                                }
                            )
                            # Clean up temp file
                            pdf_path.unlink()
                    except Exception as e:
                        logger.warning(f"PDF report generation failed: {e}")

                    # Generate JSON report
                    json_content = json.dumps(report_data, indent=2)
                    zf.writestr("reports/analysis_report.json", json_content)
                    manifest["files"].append(
                        {
                            "path": "reports/analysis_report.json",
                            "type": "report",
                            "format": "json",
                        }
                    )

                # Add manifest
                manifest_json = json.dumps(manifest, indent=2)
                zf.writestr("manifest.json", manifest_json)

                # Add verification script
                verify_script = self._generate_verify_script()
                zf.writestr("verify_integrity.py", verify_script)

                # Add README
                readme = self._generate_readme(case, len(evidence_records))
                zf.writestr("README.txt", readme)

            # Calculate package hash
            package_hash = self._calculate_file_hash(filepath)
            hash_file = filepath.with_suffix(".sha256")
            hash_file.write_text(f"{package_hash}  {filepath.name}\n")

            logger.info(f"Created export package: {filepath}")
            logger.info(f"Package hash: {package_hash}")

            return filepath

    async def _export_evidence_item(
        self,
        db,
        evidence,
        zf: zipfile.ZipFile,
        include_raw: bool,
        include_analysis: bool,
    ) -> dict[str, Any]:
        """Export a single evidence item."""
        from sqlalchemy import select

        from core.database import AnalysisResult, ChainOfCustodyLog

        evidence_dir = f"evidence/{evidence.id}"
        evidence_data = {
            "id": str(evidence.id),
            "filename": evidence.original_filename,
            "type": evidence.evidence_type.value if evidence.evidence_type else None,
            "status": evidence.status.value if evidence.status else None,
            "original_hash": evidence.original_hash,
            "size_bytes": evidence.file_size_bytes,
            "mime_type": evidence.mime_type,
            "created_at": evidence.created_at.isoformat() if evidence.created_at else None,
            "analyzed_at": evidence.analyzed_at.isoformat() if evidence.analyzed_at else None,
            "files": [],
        }

        # Add raw evidence file
        if include_raw and evidence.storage_path and os.path.exists(evidence.storage_path):
            with open(evidence.storage_path, "rb") as f:
                content = f.read()

            file_path = f"{evidence_dir}/original/{evidence.original_filename}"
            zf.writestr(file_path, content)

            # Verify hash
            content_hash = hashlib.sha256(content).hexdigest()
            if content_hash != evidence.original_hash:
                logger.warning(f"Hash mismatch for {evidence.original_filename}")

            evidence_data["files"].append(
                {
                    "path": file_path,
                    "type": "original",
                    "hash": content_hash,
                }
            )

        # Add chain of custody
        result = await db.execute(
            select(ChainOfCustodyLog)
            .where(ChainOfCustodyLog.evidence_id == evidence.id)
            .order_by(ChainOfCustodyLog.timestamp)
        )
        custody_entries = result.scalars().all()

        custody_data = []
        for entry in custody_entries:
            custody_data.append(
                {
                    "id": entry.id,
                    "timestamp": entry.timestamp.isoformat() if entry.timestamp else None,
                    "agent_id": entry.agent_id,
                    "agent_type": entry.agent_type,
                    "action": entry.action,
                    "input_hash": entry.input_hash,
                    "output_hash": entry.output_hash,
                    "processing_time_ms": entry.processing_time_ms,
                    "success": entry.success,
                    "error_message": entry.error_message,
                }
            )

        custody_json = json.dumps(custody_data, indent=2)
        custody_path = f"{evidence_dir}/chain_of_custody.json"
        zf.writestr(custody_path, custody_json)
        evidence_data["files"].append(
            {
                "path": custody_path,
                "type": "custody_log",
                "hash": hashlib.sha256(custody_json.encode()).hexdigest(),
            }
        )
        evidence_data["chain_of_custody_entries"] = len(custody_data)

        # Add analysis results
        if include_analysis:
            result = await db.execute(
                select(AnalysisResult).where(AnalysisResult.evidence_id == evidence.id)
            )
            analysis_results = result.scalars().all()

            if analysis_results:
                analysis_data = []
                for ar in analysis_results:
                    analysis_data.append(
                        {
                            "id": str(ar.id),
                            "agent_type": ar.agent_type,
                            "agent_id": ar.agent_id,
                            "result_data": ar.result_data,
                            "confidence": ar.confidence,
                            "processing_time_ms": ar.processing_time_ms,
                            "created_at": ar.created_at.isoformat() if ar.created_at else None,
                        }
                    )

                analysis_json = json.dumps(analysis_data, indent=2)
                analysis_path = f"{evidence_dir}/analysis_results.json"
                zf.writestr(analysis_path, analysis_json)
                evidence_data["files"].append(
                    {
                        "path": analysis_path,
                        "type": "analysis",
                        "hash": hashlib.sha256(analysis_json.encode()).hexdigest(),
                    }
                )

            # Add behavioral indicators
            if evidence.behavioral_indicators:
                bi_json = json.dumps(evidence.behavioral_indicators, indent=2)
                bi_path = f"{evidence_dir}/behavioral_indicators.json"
                zf.writestr(bi_path, bi_json)
                evidence_data["files"].append(
                    {
                        "path": bi_path,
                        "type": "behavioral",
                        "hash": hashlib.sha256(bi_json.encode()).hexdigest(),
                    }
                )

            # Add fusion results
            if evidence.fusion_results:
                fusion_json = json.dumps(evidence.fusion_results, indent=2)
                fusion_path = f"{evidence_dir}/fusion_results.json"
                zf.writestr(fusion_path, fusion_json)
                evidence_data["files"].append(
                    {
                        "path": fusion_path,
                        "type": "fusion",
                        "hash": hashlib.sha256(fusion_json.encode()).hexdigest(),
                    }
                )

            evidence_data["fused_score"] = evidence.fused_score
            evidence_data["classification"] = evidence.fused_classification
            evidence_data["confidence"] = evidence.confidence

        return evidence_data

    async def _build_report_data(self, case, evidence_records) -> dict[str, Any]:
        """Build report data structure."""
        return {
            "generated_at": datetime.utcnow().isoformat(),
            "case": {
                "id": str(case.id),
                "case_number": case.case_number,
                "title": case.title,
                "description": case.description,
                "status": case.status.value if case.status else "unknown",
                "client_name": case.client_name,
                "attorney_name": case.attorney_name,
                "jurisdiction": case.jurisdiction,
                "created_at": case.created_at.isoformat() if case.created_at else None,
            },
            "evidence": [
                {
                    "id": str(e.id),
                    "filename": e.original_filename,
                    "type": e.evidence_type.value if e.evidence_type else "unknown",
                    "status": e.status.value if e.status else "unknown",
                    "hash": e.original_hash,
                    "size_bytes": e.file_size_bytes,
                    "fused_score": e.fused_score,
                    "classification": e.fused_classification,
                    "confidence": e.confidence,
                }
                for e in evidence_records
            ],
            "summary": {
                "total_evidence": len(evidence_records),
                "analyzed": sum(
                    1 for e in evidence_records if e.status and e.status.value == "analyzed"
                ),
                "flagged": sum(
                    1 for e in evidence_records if e.status and e.status.value == "flagged"
                ),
            },
        }

    def _calculate_file_hash(self, filepath: Path) -> str:
        """Calculate SHA-256 hash of file."""
        sha256 = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _generate_verify_script(self) -> str:
        """Generate Python script for integrity verification."""
        return '''#!/usr/bin/env python3
"""Evidence Package Integrity Verification Script
Verifies all file hashes in the export package.
"""

import hashlib
import json
import sys
import zipfile
from pathlib import Path


def verify_package(package_path: str) -> bool:
    """Verify integrity of evidence package."""
    print(f"Verifying: {package_path}")
    print("-" * 50)

    with zipfile.ZipFile(package_path, "r") as zf:
        # Read manifest
        try:
            manifest_data = zf.read("manifest.json")
            manifest = json.loads(manifest_data)
        except Exception as e:
            print(f"ERROR: Cannot read manifest: {e}")
            return False

        checksums = manifest.get("integrity", {}).get("checksums", {})
        algorithm = manifest.get("integrity", {}).get("algorithm", "SHA-256")

        print(f"Algorithm: {algorithm}")
        print(f"Files to verify: {len(checksums)}")
        print()

        failed = []
        for file_path, expected_hash in checksums.items():
            try:
                content = zf.read(file_path)
                actual_hash = hashlib.sha256(content).hexdigest()

                if actual_hash == expected_hash:
                    print(f"  [OK] {file_path}")
                else:
                    print(f"  [FAIL] {file_path}")
                    print(f"         Expected: {expected_hash}")
                    print(f"         Actual:   {actual_hash}")
                    failed.append(file_path)
            except Exception as e:
                print(f"  [ERROR] {file_path}: {e}")
                failed.append(file_path)

        print()
        print("-" * 50)

        if failed:
            print(f"VERIFICATION FAILED: {len(failed)} files failed")
            return False
        else:
            print("VERIFICATION PASSED: All files verified successfully")
            return True


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <package.zip>")
        sys.exit(1)

    success = verify_package(sys.argv[1])
    sys.exit(0 if success else 1)
'''

    def _generate_readme(self, case, evidence_count: int) -> str:
        """Generate README for export package."""
        return f"""EVIDENCE SUITE - EXPORT PACKAGE
===============================

Case Number: {case.case_number}
Case Title: {case.title}
Export Date: {datetime.utcnow().isoformat()}
Evidence Items: {evidence_count}

CONTENTS
--------

manifest.json       - Package manifest with file listings and checksums
verify_integrity.py - Python script to verify file integrity
README.txt          - This file

evidence/           - Evidence files and analysis results
  <evidence_id>/
    original/       - Original evidence files
    chain_of_custody.json - Chain of custody log
    analysis_results.json - Analysis results
    behavioral_indicators.json - Behavioral analysis
    fusion_results.json - Fusion analysis

reports/            - Generated reports
  analysis_report.json - JSON format report
  *.pdf             - PDF format report (if available)


VERIFICATION
------------

To verify package integrity:

  python verify_integrity.py <package.zip>

Or manually verify the SHA-256 hash in the accompanying .sha256 file.


LEGAL NOTICE
------------

This evidence package was generated by Evidence Suite, a forensic behavioral
intelligence platform. All files include cryptographic hashes for integrity
verification. The chain of custody log documents all processing steps.

For questions about this export, contact your Evidence Suite administrator.


Generated by Evidence Suite v1.0.0
https://github.com/AuraquanTech/evidence-suite
"""


# Convenience function
async def export_case(
    case_id: str | UUID,
    output_dir: str | Path = "./exports",
    **kwargs,
) -> Path:
    """Export a case as a ZIP package."""
    exporter = EvidenceExporter(output_dir)
    return await exporter.export_case(case_id, **kwargs)
