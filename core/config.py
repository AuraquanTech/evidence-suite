"""
Evidence Suite - Configuration
System-wide configuration with sensible defaults.
"""
from typing import Dict, List, Optional, Literal
from pydantic import BaseModel, Field


class OCRConfig(BaseModel):
    """OCR Agent configuration."""
    engines: List[str] = ["tesseract", "easyocr"]
    tesseract_path: str = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    languages: List[str] = ["en"]
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
    darvo_keywords: List[str] = [
        "you always", "you never", "it's your fault",
        "i'm the victim", "you made me", "look what you did",
        "you're crazy", "that never happened", "you're imagining"
    ]

    # Gaslighting indicators
    gaslighting_phrases: List[str] = [
        "you're being paranoid", "that's not what happened",
        "you're too sensitive", "i never said that",
        "you're remembering it wrong", "you're overreacting"
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


class Config(BaseModel):
    """Master configuration for Evidence Suite."""
    # System settings
    project_name: str = "Evidence Suite: Savant Genesis Edition"
    version: str = "0.1.0"
    debug: bool = False

    # Ray settings
    ray_address: Optional[str] = None  # None = local mode
    num_cpus: Optional[int] = None
    num_gpus: Optional[int] = None

    # Agent configurations
    ocr: OCRConfig = Field(default_factory=OCRConfig)
    behavioral: BehavioralConfig = Field(default_factory=BehavioralConfig)
    fusion: FusionConfig = Field(default_factory=FusionConfig)
    refinement: RefinementConfig = Field(default_factory=RefinementConfig)

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
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, path: str) -> None:
        """Save configuration to YAML file."""
        import yaml
        with open(path, 'w') as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False)


# Default configuration instance
default_config = Config()
