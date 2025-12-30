"""Evidence Suite - Configuration
System-wide configuration with sensible defaults.
"""

from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class OCRConfig(BaseModel):
    """OCR Agent configuration."""

    engines: list[str] = ["tesseract", "easyocr"]
    tesseract_path: str = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    languages: list[str] = ["en"]
    confidence_threshold: float = 0.6
    preprocessing: bool = True
    deskew: bool = True


class BehavioralConfig(BaseModel):
    """Behavioral Analysis Agent configuration."""

    model_name: str = "bert-base-uncased"
    device: Literal["cuda", "cpu", "auto"] = "auto"
    max_sequence_length: int = 512
    batch_size: int = 8

    # DARVO detection keywords
    darvo_keywords: list[str] = [
        "you always",
        "you never",
        "it's your fault",
        "i'm the victim",
        "you made me",
        "look what you did",
        "you're crazy",
        "that never happened",
        "you're imagining",
    ]

    # Gaslighting indicators
    gaslighting_phrases: list[str] = [
        "you're being paranoid",
        "that's not what happened",
        "you're too sensitive",
        "i never said that",
        "you're remembering it wrong",
        "you're overreacting",
    ]


class FusionConfig(BaseModel):
    """Fusion Agent configuration."""

    strategy: Literal["hybrid_late", "weighted_average", "attention"] = "hybrid_late"
    alpha: float = 0.85  # Weight for primary modality
    confidence_threshold: float = 0.7
    require_consensus: bool = False


class RefinementConfig(BaseModel):
    """Recursive refinement configuration."""

    max_cycles: int = 5
    improvement_threshold: float = 0.03  # 3% improvement target
    rollback_on_degradation: bool = True
    ami_threshold: float = 0.85  # Adjusted Mutual Information threshold


class AudioConfig(BaseModel):
    """Audio Agent configuration."""

    whisper_model: str = "base"  # tiny, base, small, medium, large
    language: str | None = None  # None = auto-detect
    enable_diarization: bool = True
    device: Literal["cuda", "cpu", "auto"] = "auto"
    batch_size: int = 16
    word_timestamps: bool = True


class EmailConfig(BaseModel):
    """Email Agent configuration."""

    extract_attachments: bool = True
    parse_headers: bool = True
    detect_spoofing: bool = True
    extract_urls: bool = True
    max_attachment_size_mb: int = 50


class VideoConfig(BaseModel):
    """Video Agent configuration."""

    extract_audio: bool = True
    extract_frames: bool = True
    frame_interval_seconds: float = 1.0
    max_frames: int = 100
    whisper_model: str = "base"
    device: Literal["cuda", "cpu", "auto"] = "auto"
    detect_faces: bool = False  # Disabled by default for privacy
    scene_detection: bool = True
    scene_threshold: float = 30.0


class ImageConfig(BaseModel):
    """Image Agent configuration."""

    extract_exif: bool = True
    detect_manipulation: bool = True
    extract_text: bool = True  # OCR for images
    device: Literal["cuda", "cpu", "auto"] = "auto"
    hash_algorithms: list[str] = ["sha256", "md5", "phash"]
    max_image_size_mb: int = 100


class Config(BaseModel):
    """Master configuration for Evidence Suite."""

    # System settings
    project_name: str = "Evidence Suite: Savant Genesis Edition"
    version: str = "0.1.0"
    debug: bool = False

    # Ray settings
    ray_address: str | None = None  # None = local mode
    num_cpus: int | None = None
    num_gpus: int | None = None

    # Agent configurations
    ocr: OCRConfig = Field(default_factory=OCRConfig)
    behavioral: BehavioralConfig = Field(default_factory=BehavioralConfig)
    fusion: FusionConfig = Field(default_factory=FusionConfig)
    refinement: RefinementConfig = Field(default_factory=RefinementConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    email: EmailConfig = Field(default_factory=EmailConfig)
    video: VideoConfig = Field(default_factory=VideoConfig)
    image: ImageConfig = Field(default_factory=ImageConfig)

    # Storage
    event_store_url: str = "postgresql://localhost:5432/evidence_suite"
    redis_url: str = "redis://localhost:6379"

    # Logging
    log_level: str = "INFO"
    log_format: str = "{time} | {level} | {name} | {message}"

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load configuration from YAML file."""
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, path: str) -> None:
        """Save configuration to YAML file."""
        import yaml

        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False)


# Default configuration instance
default_config = Config()


class DatabaseSettings(BaseSettings):
    """PostgreSQL database configuration."""

    host: str = "localhost"
    port: int = 5432
    user: str = "postgres"
    password: str = "postgres"
    database: str = "evidence_suite"
    pool_size: int = 10
    max_overflow: int = 20

    @property
    def url(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"

    @property
    def async_url(self) -> str:
        return f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"

    class Config:
        env_prefix = "POSTGRES_"
        env_file = ".env"
        extra = "ignore"


class RedisSettings(BaseSettings):
    """Redis cache configuration."""

    host: str = "localhost"
    port: int = 6379
    password: str | None = None
    db: int = 0
    analysis_cache_ttl: int = 3600  # 1 hour
    embedding_cache_ttl: int = 86400  # 24 hours

    @property
    def url(self) -> str:
        if self.password:
            return f"redis://:{self.password}@{self.host}:{self.port}/{self.db}"
        return f"redis://{self.host}:{self.port}/{self.db}"

    class Config:
        env_prefix = "REDIS_"
        env_file = ".env"
        extra = "ignore"


class APISettings(BaseSettings):
    """FastAPI configuration."""

    title: str = "Evidence Suite API"
    version: str = "1.0.0"
    description: str = "Forensic behavioral intelligence platform"
    host: str = "0.0.0.0"
    port: int = 8000
    jwt_secret: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 30
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:8080"]

    class Config:
        env_prefix = "API_"
        env_file = ".env"
        extra = "ignore"


class HardwareSettings(BaseSettings):
    """GPU and hardware configuration."""

    thermal_warning: int = 82
    thermal_hot: int = 85
    thermal_critical: int = 90
    vram_budget_mb: int = 20480
    batch_size: int = 96
    max_workers: int = 16
    use_onnx: bool = True
    use_tensorrt: bool = False

    class Config:
        env_prefix = "HW_"
        env_file = ".env"
        extra = "ignore"


# Settings instances
db_settings = DatabaseSettings()
redis_settings = RedisSettings()
api_settings = APISettings()
hw_settings = HardwareSettings()
