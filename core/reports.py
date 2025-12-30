"""Evidence Suite - Report Generation
Generate forensic analysis reports in various formats.
"""

import io
import os
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from loguru import logger


class ReportGenerator:
    """Generate forensic analysis reports.

    Supports:
    - PDF (via reportlab)
    - HTML
    - JSON
    - Markdown
    """

    def __init__(self, output_dir: str | Path = "./reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def generate_case_report(
        self,
        case_id: str | UUID,
        format: str = "pdf",
        include_evidence: bool = True,
        include_analysis: bool = True,
    ) -> Path:
        """Generate a complete case report.

        Args:
            case_id: Case UUID
            format: Output format (pdf, html, json, md)
            include_evidence: Include evidence details
            include_analysis: Include analysis results

        Returns:
            Path to generated report
        """
        from sqlalchemy import select

        from core.database import AnalysisResult, Case, ChainOfCustodyLog, EvidenceRecord
        from core.database.session import get_async_session

        async with get_async_session() as db:
            # Get case
            result = await db.execute(select(Case).where(Case.id == UUID(str(case_id))))
            case = result.scalar_one_or_none()

            if not case:
                raise ValueError(f"Case not found: {case_id}")

            # Get evidence
            evidence_records = []
            if include_evidence:
                result = await db.execute(
                    select(EvidenceRecord).where(EvidenceRecord.case_id == case.id)
                )
                evidence_records = result.scalars().all()

            # Build report data
            report_data = {
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
                "evidence": [],
                "summary": {
                    "total_evidence": len(evidence_records),
                    "analyzed": 0,
                    "flagged": 0,
                },
            }

            for evidence in evidence_records:
                evidence_data = {
                    "id": str(evidence.id),
                    "filename": evidence.original_filename,
                    "type": evidence.evidence_type.value if evidence.evidence_type else "unknown",
                    "status": evidence.status.value if evidence.status else "unknown",
                    "hash": evidence.original_hash,
                    "size_bytes": evidence.file_size_bytes,
                    "created_at": evidence.created_at.isoformat() if evidence.created_at else None,
                }

                if include_analysis:
                    evidence_data["fused_score"] = evidence.fused_score
                    evidence_data["classification"] = evidence.fused_classification
                    evidence_data["confidence"] = evidence.confidence
                    evidence_data["behavioral_indicators"] = evidence.behavioral_indicators

                report_data["evidence"].append(evidence_data)

                if evidence.status and evidence.status.value == "analyzed":
                    report_data["summary"]["analyzed"] += 1
                if evidence.status and evidence.status.value == "flagged":
                    report_data["summary"]["flagged"] += 1

        # Generate report in requested format
        if format == "pdf":
            return self._generate_pdf(report_data, case.case_number)
        if format == "html":
            return self._generate_html(report_data, case.case_number)
        if format == "json":
            return self._generate_json(report_data, case.case_number)
        if format == "md":
            return self._generate_markdown(report_data, case.case_number)
        raise ValueError(f"Unsupported format: {format}")

    def _generate_pdf(self, data: dict[str, Any], case_number: str) -> Path:
        """Generate PDF report."""
        try:
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import letter
            from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
            from reportlab.lib.units import inch
            from reportlab.platypus import (
                Paragraph,
                SimpleDocTemplate,
                Spacer,
                Table,
                TableStyle,
            )
        except ImportError:
            logger.warning("reportlab not installed, falling back to HTML")
            return self._generate_html(data, case_number)

        filename = f"report_{case_number}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        filepath = self.output_dir / filename

        doc = SimpleDocTemplate(str(filepath), pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        # Title
        title_style = ParagraphStyle(
            "Title",
            parent=styles["Title"],
            fontSize=18,
            spaceAfter=30,
        )
        story.append(Paragraph("Evidence Suite - Case Report", title_style))
        story.append(Paragraph(f"Case: {data['case']['case_number']}", styles["Heading2"]))
        story.append(Spacer(1, 0.25 * inch))

        # Case Information
        story.append(Paragraph("Case Information", styles["Heading2"]))
        case_info = [
            ["Title:", data["case"]["title"]],
            ["Status:", data["case"]["status"]],
            ["Client:", data["case"]["client_name"] or "N/A"],
            ["Attorney:", data["case"]["attorney_name"] or "N/A"],
            ["Jurisdiction:", data["case"]["jurisdiction"] or "N/A"],
            ["Created:", data["case"]["created_at"] or "N/A"],
        ]
        table = Table(case_info, colWidths=[1.5 * inch, 4.5 * inch])
        table.setStyle(
            TableStyle(
                [
                    ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ]
            )
        )
        story.append(table)
        story.append(Spacer(1, 0.25 * inch))

        # Summary
        story.append(Paragraph("Summary", styles["Heading2"]))
        summary = data["summary"]
        summary_text = f"""
        Total Evidence Items: {summary["total_evidence"]}<br/>
        Analyzed: {summary["analyzed"]}<br/>
        Flagged: {summary["flagged"]}
        """
        story.append(Paragraph(summary_text, styles["Normal"]))
        story.append(Spacer(1, 0.25 * inch))

        # Evidence List
        if data["evidence"]:
            story.append(Paragraph("Evidence Items", styles["Heading2"]))

            for i, evidence in enumerate(data["evidence"], 1):
                story.append(Paragraph(f"{i}. {evidence['filename']}", styles["Heading3"]))

                evidence_info = [
                    ["Type:", evidence["type"]],
                    ["Status:", evidence["status"]],
                    ["Hash:", evidence["hash"][:32] + "..."],
                ]

                if evidence.get("fused_score") is not None:
                    evidence_info.append(["Analysis Score:", f"{evidence['fused_score']:.2f}"])
                if evidence.get("classification"):
                    evidence_info.append(["Classification:", evidence["classification"]])

                table = Table(evidence_info, colWidths=[1.5 * inch, 4.5 * inch])
                table.setStyle(
                    TableStyle(
                        [
                            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                            ("FONTSIZE", (0, 0), (-1, -1), 9),
                        ]
                    )
                )
                story.append(table)
                story.append(Spacer(1, 0.15 * inch))

        # Footer
        story.append(Spacer(1, 0.5 * inch))
        footer_text = (
            f"Generated: {data['generated_at']} | Evidence Suite Forensic Intelligence Platform"
        )
        story.append(Paragraph(footer_text, styles["Normal"]))

        doc.build(story)
        logger.info(f"Generated PDF report: {filepath}")
        return filepath

    def _generate_html(self, data: dict[str, Any], case_number: str) -> Path:
        """Generate HTML report."""
        filename = f"report_{case_number}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        filepath = self.output_dir / filename

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Case Report - {data["case"]["case_number"]}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        h3 {{ color: #7f8c8d; }}
        table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 10px; text-align: left; }}
        th {{ background-color: #3498db; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .summary-box {{ background: #ecf0f1; padding: 20px; border-radius: 5px; margin: 20px 0; }}
        .evidence-item {{ background: #fff; border: 1px solid #ddd; padding: 15px; margin: 10px 0; border-radius: 5px; }}
        .score {{ font-size: 24px; font-weight: bold; color: #e74c3c; }}
        .footer {{ margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #7f8c8d; font-size: 12px; }}
    </style>
</head>
<body>
    <h1>Evidence Suite - Case Report</h1>

    <h2>Case Information</h2>
    <table>
        <tr><th>Case Number</th><td>{data["case"]["case_number"]}</td></tr>
        <tr><th>Title</th><td>{data["case"]["title"]}</td></tr>
        <tr><th>Status</th><td>{data["case"]["status"]}</td></tr>
        <tr><th>Client</th><td>{data["case"]["client_name"] or "N/A"}</td></tr>
        <tr><th>Attorney</th><td>{data["case"]["attorney_name"] or "N/A"}</td></tr>
        <tr><th>Jurisdiction</th><td>{data["case"]["jurisdiction"] or "N/A"}</td></tr>
        <tr><th>Created</th><td>{data["case"]["created_at"] or "N/A"}</td></tr>
    </table>

    <div class="summary-box">
        <h2>Summary</h2>
        <p><strong>Total Evidence Items:</strong> {data["summary"]["total_evidence"]}</p>
        <p><strong>Analyzed:</strong> {data["summary"]["analyzed"]}</p>
        <p><strong>Flagged:</strong> {data["summary"]["flagged"]}</p>
    </div>

    <h2>Evidence Items</h2>
"""

        for i, evidence in enumerate(data["evidence"], 1):
            score_html = ""
            if evidence.get("fused_score") is not None:
                score_html = f'<p class="score">{evidence["fused_score"]:.2f}</p>'

            html += f"""
    <div class="evidence-item">
        <h3>{i}. {evidence["filename"]}</h3>
        <table>
            <tr><th>Type</th><td>{evidence["type"]}</td></tr>
            <tr><th>Status</th><td>{evidence["status"]}</td></tr>
            <tr><th>Hash</th><td>{evidence["hash"]}</td></tr>
            <tr><th>Size</th><td>{evidence["size_bytes"]} bytes</td></tr>
        </table>
        {score_html}
        {"<p><strong>Classification:</strong> " + evidence["classification"] + "</p>" if evidence.get("classification") else ""}
    </div>
"""

        html += f"""
    <div class="footer">
        <p>Generated: {data["generated_at"]} | Evidence Suite Forensic Intelligence Platform</p>
    </div>
</body>
</html>
"""

        filepath.write_text(html)
        logger.info(f"Generated HTML report: {filepath}")
        return filepath

    def _generate_json(self, data: dict[str, Any], case_number: str) -> Path:
        """Generate JSON report."""
        import json

        filename = f"report_{case_number}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.output_dir / filename

        filepath.write_text(json.dumps(data, indent=2))
        logger.info(f"Generated JSON report: {filepath}")
        return filepath

    def _generate_markdown(self, data: dict[str, Any], case_number: str) -> Path:
        """Generate Markdown report."""
        filename = f"report_{case_number}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        filepath = self.output_dir / filename

        md = f"""# Evidence Suite - Case Report

## Case Information

| Field | Value |
|-------|-------|
| Case Number | {data["case"]["case_number"]} |
| Title | {data["case"]["title"]} |
| Status | {data["case"]["status"]} |
| Client | {data["case"]["client_name"] or "N/A"} |
| Attorney | {data["case"]["attorney_name"] or "N/A"} |
| Jurisdiction | {data["case"]["jurisdiction"] or "N/A"} |
| Created | {data["case"]["created_at"] or "N/A"} |

## Summary

- **Total Evidence Items:** {data["summary"]["total_evidence"]}
- **Analyzed:** {data["summary"]["analyzed"]}
- **Flagged:** {data["summary"]["flagged"]}

## Evidence Items

"""

        for i, evidence in enumerate(data["evidence"], 1):
            md += f"""### {i}. {evidence["filename"]}

| Field | Value |
|-------|-------|
| Type | {evidence["type"]} |
| Status | {evidence["status"]} |
| Hash | `{evidence["hash"]}` |
| Size | {evidence["size_bytes"]} bytes |
"""
            if evidence.get("fused_score") is not None:
                md += f"| Analysis Score | **{evidence['fused_score']:.2f}** |\n"
            if evidence.get("classification"):
                md += f"| Classification | {evidence['classification']} |\n"
            md += "\n"

        md += f"""---

*Generated: {data["generated_at"]} | Evidence Suite Forensic Intelligence Platform*
"""

        filepath.write_text(md)
        logger.info(f"Generated Markdown report: {filepath}")
        return filepath


# Convenience function
async def generate_report(
    case_id: str | UUID,
    format: str = "pdf",
    output_dir: str | Path = "./reports",
) -> Path:
    """Generate a case report."""
    generator = ReportGenerator(output_dir)
    return await generator.generate_case_report(case_id, format)
