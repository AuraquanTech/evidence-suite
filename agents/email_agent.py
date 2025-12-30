"""Evidence Suite - Email Agent
Email parsing and forensic analysis for message evidence.
"""

from __future__ import annotations

import email
import hashlib
import re
from email import policy
from email.parser import BytesParser
from typing import Any

from loguru import logger

from agents.base import BaseAgent
from core.models import AnalysisResult, EvidencePacket, EvidenceType, ProcessingStage


class EmailConfig:
    """Email Agent configuration."""

    def __init__(
        self,
        extract_attachments: bool = True,
        parse_headers: bool = True,
        detect_spoofing: bool = True,
        extract_urls: bool = True,
        max_attachment_size_mb: int = 50,
    ):
        self.extract_attachments = extract_attachments
        self.parse_headers = parse_headers
        self.detect_spoofing = detect_spoofing
        self.extract_urls = extract_urls
        self.max_attachment_size_mb = max_attachment_size_mb


class EmailAgent(BaseAgent):
    """Sensory layer agent for email forensic analysis.

    Features:
    - Email header parsing and analysis
    - Spoofing detection (SPF, DKIM, DMARC)
    - Attachment extraction and hashing
    - URL extraction and analysis
    - Thread reconstruction
    - Timestamp analysis
    """

    def __init__(self, agent_id: str | None = None, config: EmailConfig | None = None):
        self.email_config = config or EmailConfig()
        super().__init__(
            agent_id=agent_id,
            agent_type="email",
            config={
                "extract_attachments": self.email_config.extract_attachments,
                "parse_headers": self.email_config.parse_headers,
                "detect_spoofing": self.email_config.detect_spoofing,
            },
        )

    async def _setup(self) -> None:
        """Initialize email processing."""
        logger.info("Email Agent initialized")

    async def _process_impl(self, packet: EvidencePacket) -> EvidencePacket:
        """Process email evidence.

        Steps:
        1. Parse email structure
        2. Extract headers
        3. Extract body (plain text and HTML)
        4. Extract and hash attachments
        5. Detect spoofing indicators
        6. Extract URLs
        """
        if packet.evidence_type != EvidenceType.EMAIL:
            raise ValueError(f"Expected email evidence, got {packet.evidence_type}")

        if not packet.raw_content:
            raise ValueError("No email content to process")

        # Parse email
        msg = BytesParser(policy=policy.default).parsebytes(packet.raw_content)

        # Extract components
        headers = self._extract_headers(msg)
        body_text, body_html = self._extract_body(msg)
        attachments = (
            self._extract_attachments(msg) if self.email_config.extract_attachments else []
        )
        urls = self._extract_urls(body_text + body_html) if self.email_config.extract_urls else []
        spoofing_indicators = (
            self._detect_spoofing(msg, headers) if self.email_config.detect_spoofing else {}
        )

        # Thread analysis
        thread_info = self._analyze_thread(msg, headers)

        # Build full text for behavioral analysis
        full_text = self._build_full_text(headers, body_text)

        # Create analysis result
        analysis = AnalysisResult(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            confidence=self._calculate_confidence(headers, spoofing_indicators),
            findings={
                "from": headers.get("from"),
                "to": headers.get("to"),
                "subject": headers.get("subject"),
                "date": headers.get("date"),
                "attachment_count": len(attachments),
                "url_count": len(urls),
                "spoofing_risk": spoofing_indicators.get("risk_level", "unknown"),
                "is_reply": thread_info.get("is_reply", False),
                "thread_depth": thread_info.get("depth", 0),
            },
            raw_output={
                "headers": headers,
                "attachments": attachments,
                "urls": urls,
                "spoofing_indicators": spoofing_indicators,
                "thread_info": thread_info,
            },
        )

        return packet.with_updates(
            extracted_text=full_text,
            stage=ProcessingStage.OCR_PROCESSED,
            analysis_results=packet.analysis_results + [analysis],
            source_metadata={
                **packet.source_metadata,
                "email_parsed": {
                    "headers": headers,
                    "attachments": [a["filename"] for a in attachments],
                    "urls": urls[:20],  # Limit stored URLs
                },
            },
        )

    def _extract_headers(self, msg: email.message.EmailMessage) -> dict[str, Any]:
        """Extract and parse email headers."""
        headers = {}

        # Standard headers
        for header in [
            "From",
            "To",
            "Cc",
            "Bcc",
            "Subject",
            "Date",
            "Message-ID",
            "In-Reply-To",
            "References",
        ]:
            value = msg.get(header)
            if value:
                headers[header.lower().replace("-", "_")] = str(value)

        # Parse date
        if "date" in headers:
            try:
                headers["date_parsed"] = email.utils.parsedate_to_datetime(
                    headers["date"]
                ).isoformat()
            except Exception:
                pass

        # Extract email addresses
        if "from" in headers:
            headers["from_address"] = self._extract_email_address(headers["from"])
        if "to" in headers:
            headers["to_addresses"] = [
                self._extract_email_address(addr) for addr in headers["to"].split(",")
            ]

        # Authentication headers
        for auth_header in [
            "Received-SPF",
            "Authentication-Results",
            "DKIM-Signature",
            "ARC-Authentication-Results",
        ]:
            value = msg.get(auth_header)
            if value:
                headers[auth_header.lower().replace("-", "_")] = str(value)

        # Received chain (for routing analysis)
        received = msg.get_all("Received", [])
        headers["received_chain"] = [str(r) for r in received]

        return headers

    def _extract_email_address(self, header_value: str) -> str | None:
        """Extract email address from header value."""
        match = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", header_value)
        return match.group(0) if match else None

    def _extract_body(self, msg: email.message.EmailMessage) -> tuple[str, str]:
        """Extract plain text and HTML body."""
        text_body = ""
        html_body = ""

        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                try:
                    payload = part.get_payload(decode=True)
                    if payload:
                        charset = part.get_content_charset() or "utf-8"
                        decoded = payload.decode(charset, errors="ignore")

                        if content_type == "text/plain":
                            text_body += decoded
                        elif content_type == "text/html":
                            html_body += decoded
                except Exception as e:
                    logger.warning(f"Failed to decode email part: {e}")
        else:
            try:
                payload = msg.get_payload(decode=True)
                if payload:
                    charset = msg.get_content_charset() or "utf-8"
                    decoded = payload.decode(charset, errors="ignore")
                    content_type = msg.get_content_type()

                    if content_type == "text/plain":
                        text_body = decoded
                    elif content_type == "text/html":
                        html_body = decoded
            except Exception as e:
                logger.warning(f"Failed to decode email body: {e}")

        # If only HTML, extract text from it
        if not text_body and html_body:
            text_body = self._html_to_text(html_body)

        return text_body, html_body

    def _html_to_text(self, html: str) -> str:
        """Convert HTML to plain text."""
        # Simple HTML to text conversion
        # Remove script and style elements
        text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)

        # Replace common HTML entities
        text = text.replace("&nbsp;", " ")
        text = text.replace("&lt;", "<")
        text = text.replace("&gt;", ">")
        text = text.replace("&amp;", "&")

        # Remove tags
        text = re.sub(r"<[^>]+>", " ", text)

        # Clean up whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def _extract_attachments(self, msg: email.message.EmailMessage) -> list[dict[str, Any]]:
        """Extract and hash attachments."""
        attachments = []

        for part in msg.walk():
            content_disposition = part.get("Content-Disposition", "")
            if "attachment" in content_disposition or "inline" in content_disposition:
                filename = part.get_filename()
                if filename:
                    try:
                        payload = part.get_payload(decode=True)
                        if payload:
                            size = len(payload)
                            max_size = self.email_config.max_attachment_size_mb * 1024 * 1024

                            if size <= max_size:
                                file_hash = hashlib.sha256(payload).hexdigest()
                            else:
                                file_hash = "size_exceeded"

                            attachments.append(
                                {
                                    "filename": filename,
                                    "content_type": part.get_content_type(),
                                    "size_bytes": size,
                                    "sha256": file_hash,
                                }
                            )
                    except Exception as e:
                        logger.warning(f"Failed to process attachment {filename}: {e}")

        return attachments

    def _extract_urls(self, text: str) -> list[str]:
        """Extract URLs from text."""
        url_pattern = r'https?://[^\s<>"\']+|www\.[^\s<>"\']+'
        urls = re.findall(url_pattern, text, re.IGNORECASE)

        # Clean up URLs
        cleaned = []
        for url in urls:
            # Remove trailing punctuation
            url = re.sub(r"[.,;:!?)]+$", "", url)
            if url not in cleaned:
                cleaned.append(url)

        return cleaned

    def _detect_spoofing(
        self, msg: email.message.EmailMessage, headers: dict[str, Any]
    ) -> dict[str, Any]:
        """Detect email spoofing indicators."""
        indicators = {
            "spf_result": None,
            "dkim_result": None,
            "dmarc_result": None,
            "reply_to_mismatch": False,
            "display_name_mismatch": False,
            "suspicious_routing": False,
            "risk_level": "low",
            "warnings": [],
        }

        # Check SPF
        spf = headers.get("received_spf", "")
        if "pass" in spf.lower():
            indicators["spf_result"] = "pass"
        elif "fail" in spf.lower():
            indicators["spf_result"] = "fail"
            indicators["warnings"].append("SPF check failed")
        elif "softfail" in spf.lower():
            indicators["spf_result"] = "softfail"
            indicators["warnings"].append("SPF softfail")

        # Check Authentication-Results for DKIM and DMARC
        auth_results = headers.get("authentication_results", "")
        if "dkim=pass" in auth_results.lower():
            indicators["dkim_result"] = "pass"
        elif "dkim=fail" in auth_results.lower():
            indicators["dkim_result"] = "fail"
            indicators["warnings"].append("DKIM check failed")

        if "dmarc=pass" in auth_results.lower():
            indicators["dmarc_result"] = "pass"
        elif "dmarc=fail" in auth_results.lower():
            indicators["dmarc_result"] = "fail"
            indicators["warnings"].append("DMARC check failed")

        # Check Reply-To mismatch
        reply_to = msg.get("Reply-To")
        from_addr = headers.get("from_address")
        if reply_to and from_addr:
            reply_to_addr = self._extract_email_address(reply_to)
            if reply_to_addr and reply_to_addr != from_addr:
                indicators["reply_to_mismatch"] = True
                indicators["warnings"].append(
                    f"Reply-To ({reply_to_addr}) differs from From ({from_addr})"
                )

        # Check display name vs email domain
        from_header = headers.get("from", "")
        if from_addr:
            domain = from_addr.split("@")[-1] if "@" in from_addr else ""
            # Check if display name contains a different domain
            domain_pattern = r"[\w\.-]+\.(com|org|net|gov|edu|io|co)"
            display_domains = re.findall(domain_pattern, from_header, re.IGNORECASE)
            if display_domains and domain.lower() not in from_header.lower():
                indicators["display_name_mismatch"] = True
                indicators["warnings"].append("Display name may contain misleading domain")

        # Analyze routing
        received_chain = headers.get("received_chain", [])
        if len(received_chain) > 10:
            indicators["suspicious_routing"] = True
            indicators["warnings"].append(f"Unusual routing chain ({len(received_chain)} hops)")

        # Calculate risk level
        warning_count = len(indicators["warnings"])
        if warning_count >= 3:
            indicators["risk_level"] = "high"
        elif warning_count >= 1:
            indicators["risk_level"] = "medium"
        else:
            indicators["risk_level"] = "low"

        return indicators

    def _analyze_thread(
        self, msg: email.message.EmailMessage, headers: dict[str, Any]
    ) -> dict[str, Any]:
        """Analyze email thread context."""
        thread_info = {
            "is_reply": False,
            "is_forward": False,
            "depth": 0,
            "thread_id": None,
        }

        # Check In-Reply-To and References
        in_reply_to = headers.get("in_reply_to")
        references = headers.get("references")

        if in_reply_to or references:
            thread_info["is_reply"] = True

        # Count references to estimate thread depth
        if references:
            ref_ids = references.split()
            thread_info["depth"] = len(ref_ids)
            if ref_ids:
                thread_info["thread_id"] = ref_ids[0]

        # Check subject for Fwd: or Re:
        subject = headers.get("subject", "")
        if subject:
            if re.match(r"^(Fwd|Fw):", subject, re.IGNORECASE):
                thread_info["is_forward"] = True
            if re.match(r"^Re:", subject, re.IGNORECASE):
                thread_info["is_reply"] = True

        return thread_info

    def _build_full_text(self, headers: dict[str, Any], body: str) -> str:
        """Build full text for behavioral analysis."""
        parts = []

        # Include subject
        subject = headers.get("subject")
        if subject:
            parts.append(f"Subject: {subject}")

        # Include body
        if body:
            parts.append(body)

        return "\n\n".join(parts)

    def _calculate_confidence(
        self, headers: dict[str, Any], spoofing_indicators: dict[str, Any]
    ) -> float:
        """Calculate parsing confidence."""
        confidence = 0.8  # Base confidence

        # Boost for complete headers
        required_headers = ["from", "to", "subject", "date"]
        present = sum(1 for h in required_headers if headers.get(h))
        confidence += (present / len(required_headers)) * 0.1

        # Boost for passing authentication
        if spoofing_indicators.get("spf_result") == "pass":
            confidence += 0.03
        if spoofing_indicators.get("dkim_result") == "pass":
            confidence += 0.03
        if spoofing_indicators.get("dmarc_result") == "pass":
            confidence += 0.04

        return min(1.0, confidence)

    async def self_critique(self, result: AnalysisResult) -> float:
        """Self-critique email parsing quality."""
        if not result.is_successful:
            return 0.0

        score = result.confidence

        # Penalize high spoofing risk
        risk_level = result.findings.get("spoofing_risk", "unknown")
        if risk_level == "high":
            score *= 0.7
        elif risk_level == "medium":
            score *= 0.85

        return score
