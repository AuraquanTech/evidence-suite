"""Evidence Suite - Evidence Encryption
AES-256-GCM encryption for evidence at rest.
"""

import base64
import os
import secrets
from pathlib import Path
from typing import BinaryIO

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from loguru import logger


# Encryption settings
ENCRYPTION_KEY_ENV = "EVIDENCE_ENCRYPTION_KEY"
NONCE_SIZE = 12  # 96 bits for GCM
TAG_SIZE = 16  # 128-bit authentication tag


class EncryptionError(Exception):
    """Encryption operation failed."""

    pass


class DecryptionError(Exception):
    """Decryption operation failed."""

    pass


def get_encryption_key() -> bytes:
    """Get encryption key from environment.

    Returns:
        32-byte encryption key

    Raises:
        ValueError: If key is not configured or invalid
    """
    key_b64 = os.getenv(ENCRYPTION_KEY_ENV)

    if not key_b64:
        raise ValueError(
            f"Encryption key not configured. Set {ENCRYPTION_KEY_ENV} environment variable. "
            'Generate with: python -c "import secrets; import base64; print(base64.b64encode(secrets.token_bytes(32)).decode())"'
        )

    try:
        key = base64.b64decode(key_b64)
        if len(key) != 32:
            raise ValueError(f"Encryption key must be 32 bytes, got {len(key)}")  # noqa: TRY301
        return key
    except Exception as e:
        raise ValueError(f"Invalid encryption key: {e}") from e


def generate_key() -> str:
    """Generate a new encryption key.

    Returns:
        Base64-encoded 32-byte key
    """
    key = secrets.token_bytes(32)
    return base64.b64encode(key).decode()


class EvidenceEncryptor:
    """AES-256-GCM encryptor for evidence files.

    Features:
    - AES-256-GCM authenticated encryption
    - Unique nonce per encryption
    - File metadata encryption
    - Streaming support for large files
    """

    def __init__(self, key: bytes | None = None):
        """Initialize encryptor.

        Args:
            key: 32-byte encryption key. If None, loads from environment.
        """
        self._key = key or get_encryption_key()
        self._aesgcm = AESGCM(self._key)

    def encrypt(
        self,
        plaintext: bytes,
        associated_data: bytes | None = None,
    ) -> bytes:
        """Encrypt data.

        Args:
            plaintext: Data to encrypt
            associated_data: Optional AAD for authentication

        Returns:
            nonce + ciphertext (nonce is prepended)
        """
        nonce = secrets.token_bytes(NONCE_SIZE)

        try:
            ciphertext = self._aesgcm.encrypt(nonce, plaintext, associated_data)
            return nonce + ciphertext
        except Exception as e:
            raise EncryptionError(f"Encryption failed: {e}") from e

    def decrypt(
        self,
        ciphertext: bytes,
        associated_data: bytes | None = None,
    ) -> bytes:
        """Decrypt data.

        Args:
            ciphertext: nonce + encrypted data
            associated_data: Optional AAD used during encryption

        Returns:
            Decrypted plaintext
        """
        if len(ciphertext) < NONCE_SIZE + TAG_SIZE:
            raise DecryptionError("Ciphertext too short")

        nonce = ciphertext[:NONCE_SIZE]
        encrypted = ciphertext[NONCE_SIZE:]

        try:
            return self._aesgcm.decrypt(nonce, encrypted, associated_data)
        except Exception as e:
            raise DecryptionError(f"Decryption failed: {e}") from e

    def encrypt_file(
        self,
        input_path: str | Path,
        output_path: str | Path | None = None,
        associated_data: bytes | None = None,
        chunk_size: int = 64 * 1024,  # 64KB chunks
    ) -> Path:
        """Encrypt a file.

        For large files, reads in chunks but encrypts as a single operation.
        For truly streaming encryption of huge files, use encrypt_file_streaming.

        Args:
            input_path: Path to file to encrypt
            output_path: Output path (default: input_path + .enc)
            associated_data: Optional AAD
            chunk_size: Read chunk size

        Returns:
            Path to encrypted file
        """
        input_path = Path(input_path)
        output_path = (
            Path(output_path) if output_path else input_path.with_suffix(input_path.suffix + ".enc")
        )

        # Read file
        with open(input_path, "rb") as f:
            plaintext = f.read()

        # Encrypt
        ciphertext = self.encrypt(plaintext, associated_data)

        # Write encrypted file
        with open(output_path, "wb") as f:
            f.write(ciphertext)

        logger.debug(f"Encrypted {input_path} -> {output_path}")
        return output_path

    def decrypt_file(
        self,
        input_path: str | Path,
        output_path: str | Path | None = None,
        associated_data: bytes | None = None,
    ) -> Path:
        """Decrypt a file.

        Args:
            input_path: Path to encrypted file
            output_path: Output path (default: removes .enc suffix)
            associated_data: Optional AAD used during encryption

        Returns:
            Path to decrypted file
        """
        input_path = Path(input_path)

        if output_path:
            output_path = Path(output_path)
        elif input_path.suffix == ".enc":
            output_path = input_path.with_suffix("")
        else:
            output_path = input_path.with_suffix(".dec")

        # Read encrypted file
        with open(input_path, "rb") as f:
            ciphertext = f.read()

        # Decrypt
        plaintext = self.decrypt(ciphertext, associated_data)

        # Write decrypted file
        with open(output_path, "wb") as f:
            f.write(plaintext)

        logger.debug(f"Decrypted {input_path} -> {output_path}")
        return output_path

    def encrypt_file_streaming(
        self,
        input_file: BinaryIO,
        output_file: BinaryIO,
        chunk_size: int = 1024 * 1024,  # 1MB chunks
    ) -> None:
        """Encrypt a file using streaming (chunked) encryption.

        Each chunk is encrypted separately with its own nonce.
        Header format: [4 bytes: chunk_size][4 bytes: total_chunks]

        Args:
            input_file: Input file handle
            output_file: Output file handle
            chunk_size: Size of each chunk
        """
        import struct

        # Calculate total chunks
        input_file.seek(0, 2)  # Seek to end
        file_size = input_file.tell()
        input_file.seek(0)  # Seek back to start

        total_chunks = (file_size + chunk_size - 1) // chunk_size

        # Write header
        header = struct.pack("<II", chunk_size, total_chunks)
        output_file.write(header)

        # Encrypt chunks
        chunk_num = 0
        while True:
            chunk = input_file.read(chunk_size)
            if not chunk:
                break

            # Use chunk number as AAD for ordering protection
            aad = struct.pack("<I", chunk_num)
            encrypted_chunk = self.encrypt(chunk, aad)

            # Write chunk length and data
            output_file.write(struct.pack("<I", len(encrypted_chunk)))
            output_file.write(encrypted_chunk)

            chunk_num += 1

    def decrypt_file_streaming(
        self,
        input_file: BinaryIO,
        output_file: BinaryIO,
    ) -> None:
        """Decrypt a streaming-encrypted file.

        Args:
            input_file: Input encrypted file handle
            output_file: Output file handle
        """
        import struct

        # Read header
        header = input_file.read(8)
        if len(header) != 8:
            raise DecryptionError("Invalid file header")

        chunk_size, total_chunks = struct.unpack("<II", header)

        # Decrypt chunks
        for chunk_num in range(total_chunks):
            # Read chunk length
            length_bytes = input_file.read(4)
            if len(length_bytes) != 4:
                raise DecryptionError(f"Unexpected end of file at chunk {chunk_num}")

            chunk_length = struct.unpack("<I", length_bytes)[0]

            # Read encrypted chunk
            encrypted_chunk = input_file.read(chunk_length)
            if len(encrypted_chunk) != chunk_length:
                raise DecryptionError(f"Incomplete chunk {chunk_num}")

            # Decrypt with AAD
            aad = struct.pack("<I", chunk_num)
            plaintext = self.decrypt(encrypted_chunk, aad)

            output_file.write(plaintext)


# Convenience functions
_encryptor: EvidenceEncryptor | None = None


def get_encryptor() -> EvidenceEncryptor:
    """Get or create singleton encryptor."""
    global _encryptor
    if _encryptor is None:
        _encryptor = EvidenceEncryptor()
    return _encryptor


def encrypt_evidence(data: bytes, evidence_id: str) -> bytes:
    """Encrypt evidence data.

    Args:
        data: Raw evidence bytes
        evidence_id: Evidence UUID (used as AAD)

    Returns:
        Encrypted bytes
    """
    encryptor = get_encryptor()
    return encryptor.encrypt(data, evidence_id.encode())


def decrypt_evidence(data: bytes, evidence_id: str) -> bytes:
    """Decrypt evidence data.

    Args:
        data: Encrypted evidence bytes
        evidence_id: Evidence UUID (used as AAD)

    Returns:
        Decrypted bytes
    """
    encryptor = get_encryptor()
    return encryptor.decrypt(data, evidence_id.encode())
