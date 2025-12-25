"""Evidence Suite Agents Module"""
from .base import BaseAgent
from .ocr_agent import OCRAgent
from .behavioral_agent import BehavioralAgent
from .fusion_agent import FusionAgent
from .audio_agent import AudioAgent, AudioConfig
from .email_agent import EmailAgent, EmailConfig

__all__ = [
    "BaseAgent",
    "OCRAgent",
    "BehavioralAgent",
    "FusionAgent",
    "AudioAgent",
    "AudioConfig",
    "EmailAgent",
    "EmailConfig",
]
