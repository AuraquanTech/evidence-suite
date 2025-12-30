"""Evidence Suite - Security Utilities
Token blacklist, file validation, and security helpers.
"""

import hashlib
import mimetypes
import os
import re
import time
from pathlib import Path
from typing import BinaryIO

from loguru import logger


# Allowed MIME types for evidence uploads
ALLOWED_MIME_TYPES = {
    # Documents
    "application/pdf",
    "application/msword",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.ms-excel",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "text/plain",
    "text/csv",
    "text/html",
    "application/rtf",
    # Images
    "image/jpeg",
    "image/png",
    "image/gif",
    "image/tiff",
    "image/bmp",
    "image/webp",
    # Audio
    "audio/mpeg",
    "audio/wav",
    "audio/ogg",
    "audio/flac",
    "audio/mp4",
    "audio/x-m4a",
    # Video
    "video/mp4",
    "video/mpeg",
    "video/quicktime",
    "video/x-msvideo",
    "video/webm",
    # Email
    "message/rfc822",
    "application/mbox",
    # Archives (for multi-file evidence)
    "application/zip",
    "application/x-tar",
    "application/gzip",
}

# File extension whitelist
ALLOWED_EXTENSIONS = {
    # Documents
    ".pdf",
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
    ".txt",
    ".csv",
    ".html",
    ".rtf",
    # Images
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".tiff",
    ".tif",
    ".bmp",
    ".webp",
    # Audio
    ".mp3",
    ".wav",
    ".ogg",
    ".flac",
    ".m4a",
    ".aac",
    # Video
    ".mp4",
    ".mpeg",
    ".mpg",
    ".mov",
    ".avi",
    ".webm",
    ".mkv",
    # Email
    ".eml",
    ".mbox",
    ".msg",
    # Archives
    ".zip",
    ".tar",
    ".gz",
    ".tar.gz",
}

# Maximum file sizes (in bytes)
MAX_FILE_SIZE = int(os.getenv("MAX_UPLOAD_SIZE_MB", "500")) * 1024 * 1024  # Default 500MB
MAX_IMAGE_SIZE = 100 * 1024 * 1024  # 100MB for images
MAX_DOCUMENT_SIZE = 50 * 1024 * 1024  # 50MB for documents

# Dangerous patterns in filenames
DANGEROUS_PATTERNS = [
    r"\.\.",  # Path traversal
    r"[<>:\"|?*]",  # Windows reserved characters
    r"[\x00-\x1f]",  # Control characters
    r"^(con|prn|aux|nul|com[1-9]|lpt[1-9])(\.|$)",  # Windows reserved names
]


class TokenBlacklist:
    """In-memory token blacklist with TTL.

    In production, use Redis for distributed blacklist.
    """

    def __init__(self):
        self._blacklist: dict[str, float] = {}  # token_hash -> expiry_time
        self._cleanup_interval = 300  # 5 minutes
        self._last_cleanup = time.time()

    def add(self, token: str, ttl: int = 3600) -> None:
        """Add token to blacklist.

        Args:
            token: JWT token to blacklist
            ttl: Time to live in seconds (default 1 hour)
        """
        token_hash = self._hash_token(token)
        expiry = time.time() + ttl
        self._blacklist[token_hash] = expiry
        self._maybe_cleanup()

    def is_blacklisted(self, token: str) -> bool:
        """Check if token is blacklisted.

        Args:
            token: JWT token to check

        Returns:
            True if token is blacklisted
        """
        self._maybe_cleanup()
        token_hash = self._hash_token(token)

        if token_hash not in self._blacklist:
            return False

        # Check if expired
        if time.time() > self._blacklist[token_hash]:
            del self._blacklist[token_hash]
            return False

        return True

    def _hash_token(self, token: str) -> str:
        """Hash token for storage."""
        return hashlib.sha256(token.encode()).hexdigest()

    def _maybe_cleanup(self) -> None:
        """Periodically cleanup expired entries."""
        now = time.time()
        if now - self._last_cleanup < self._cleanup_interval:
            return

        self._last_cleanup = now
        expired = [k for k, v in self._blacklist.items() if now > v]
        for k in expired:
            del self._blacklist[k]

        if expired:
            logger.debug(f"Cleaned up {len(expired)} expired blacklist entries")


# Global blacklist instance
_token_blacklist = TokenBlacklist()


def blacklist_token(token: str, ttl: int = 3600) -> None:
    """Add token to blacklist."""
    _token_blacklist.add(token, ttl)


def is_token_blacklisted(token: str) -> bool:
    """Check if token is blacklisted."""
    return _token_blacklist.is_blacklisted(token)


class FileValidator:
    """Validates uploaded files for security."""

    def __init__(
        self,
        max_size: int = MAX_FILE_SIZE,
        allowed_types: set[str] | None = None,
        allowed_extensions: set[str] | None = None,
    ):
        self.max_size = max_size
        self.allowed_types = allowed_types or ALLOWED_MIME_TYPES
        self.allowed_extensions = allowed_extensions or ALLOWED_EXTENSIONS

    def validate(
        self,
        filename: str,
        content_type: str | None,
        file_size: int,
        content: bytes | None = None,
    ) -> tuple[bool, str | None]:
        """Validate a file upload.

        Args:
            filename: Original filename
            content_type: MIME type from upload
            file_size: Size in bytes
            content: Optional file content for magic byte validation

        Returns:
            (is_valid, error_message)
        """
        # Check filename for dangerous patterns
        error = self._validate_filename(filename)
        if error:
            return False, error

        # Check extension
        error = self._validate_extension(filename)
        if error:
            return False, error

        # Check size
        if file_size > self.max_size:
            max_mb = self.max_size / (1024 * 1024)
            return False, f"File size exceeds maximum allowed ({max_mb:.0f}MB)"

        # Check MIME type
        if content_type:
            error = self._validate_mime_type(content_type, filename)
            if error:
                return False, error

        # Validate magic bytes if content provided
        if content:
            error = self._validate_magic_bytes(content, filename)
            if error:
                return False, error

        return True, None

    def _validate_filename(self, filename: str) -> str | None:
        """Check filename for dangerous patterns."""
        for pattern in DANGEROUS_PATTERNS:
            if re.search(pattern, filename, re.IGNORECASE):
                return "Filename contains invalid characters or patterns"

        # Check for excessively long filenames
        if len(filename) > 255:
            return "Filename too long (max 255 characters)"

        return None

    def _validate_extension(self, filename: str) -> str | None:
        """Validate file extension."""
        ext = Path(filename).suffix.lower()

        if not ext:
            return "File must have an extension"

        if ext not in self.allowed_extensions:
            return f"File type '{ext}' is not allowed"

        return None

    def _validate_mime_type(self, content_type: str, filename: str) -> str | None:
        """Validate MIME type matches extension."""
        if content_type not in self.allowed_types:
            return f"Content type '{content_type}' is not allowed"

        # Cross-check MIME type with extension
        ext = Path(filename).suffix.lower()
        expected_types = mimetypes.guess_type(filename)[0]

        # Some flexibility for common mismatches
        if expected_types and content_type != expected_types:
            # Allow some common variations
            if not self._is_compatible_type(content_type, ext):
                logger.warning(f"MIME type mismatch: {content_type} for extension {ext}")
                # Don't reject, just warn - browsers can be inconsistent

        return None

    def _is_compatible_type(self, mime_type: str, extension: str) -> bool:
        """Check if MIME type is compatible with extension."""
        # Common compatible pairs
        compatible = {
            ".jpg": {"image/jpeg", "image/pjpeg"},
            ".jpeg": {"image/jpeg", "image/pjpeg"},
            ".mp3": {"audio/mpeg", "audio/mp3"},
            ".mp4": {"video/mp4", "audio/mp4"},
            ".m4a": {"audio/mp4", "audio/x-m4a", "audio/m4a"},
        }

        if extension in compatible:
            return mime_type in compatible[extension]

        return True  # Allow if no specific rule

    def _validate_magic_bytes(self, content: bytes, filename: str) -> str | None:
        """Validate file content magic bytes."""
        if len(content) < 8:
            return None  # Too short to validate

        # Magic byte signatures
        signatures = {
            b"%PDF": {".pdf"},
            b"\x89PNG": {".png"},
            b"\xff\xd8\xff": {".jpg", ".jpeg"},
            b"GIF87a": {".gif"},
            b"GIF89a": {".gif"},
            b"PK\x03\x04": {".zip", ".docx", ".xlsx", ".pptx"},
            b"ID3": {".mp3"},
            b"\xff\xfb": {".mp3"},
            b"\xff\xfa": {".mp3"},
            b"RIFF": {".wav", ".avi"},
            b"\x00\x00\x00\x18ftypmp4": {".mp4", ".m4a"},
            b"\x00\x00\x00\x1cftyp": {".mp4", ".m4a", ".mov"},
            b"\x00\x00\x00\x20ftyp": {".mp4", ".m4a"},
        }

        ext = Path(filename).suffix.lower()

        for magic, extensions in signatures.items():
            if content.startswith(magic):
                if ext not in extensions:
                    # Extension doesn't match magic bytes
                    return f"File content does not match extension '{ext}'"
                return None

        # No signature match - allow (might be text file or unknown format)
        return None


def sanitize_filename(filename: str) -> str:
    """Sanitize a filename for safe storage.

    Args:
        filename: Original filename

    Returns:
        Sanitized filename safe for filesystem storage
    """
    # Remove path components
    filename = os.path.basename(filename)

    # Remove dangerous characters
    filename = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", filename)

    # Remove leading/trailing dots and spaces
    filename = filename.strip(". ")

    # Limit length
    if len(filename) > 200:
        ext = Path(filename).suffix
        name = Path(filename).stem[: 200 - len(ext)]
        filename = name + ext

    # Ensure not empty
    if not filename:
        filename = "unnamed_file"

    return filename


def generate_safe_storage_path(
    base_dir: str,
    case_id: str,
    file_hash: str,
    filename: str,
) -> str:
    """Generate a safe storage path for evidence.

    Args:
        base_dir: Base storage directory
        case_id: Case UUID
        file_hash: SHA-256 hash of file
        filename: Original filename

    Returns:
        Safe absolute path for storage
    """
    # Validate case_id format (should be UUID)
    case_id = re.sub(r"[^a-fA-F0-9-]", "", case_id)

    # Validate file_hash (should be hex)
    file_hash = re.sub(r"[^a-fA-F0-9]", "", file_hash)

    # Sanitize filename
    safe_filename = sanitize_filename(filename)

    # Build path
    storage_path = os.path.join(
        base_dir,
        case_id,
        f"{file_hash[:16]}_{safe_filename}",
    )

    # Resolve to absolute path and verify it's under base_dir
    abs_path = os.path.abspath(storage_path)
    abs_base = os.path.abspath(base_dir)

    if not abs_path.startswith(abs_base):
        raise ValueError("Path traversal detected")

    return abs_path


# Default validator instance
file_validator = FileValidator()
