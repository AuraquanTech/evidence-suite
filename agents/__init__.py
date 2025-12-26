"""Evidence Suite Agents Module"""

from .audio_agent import AudioAgent, AudioConfig
from .base import BaseAgent
from .behavioral_agent import BehavioralAgent
from .email_agent import EmailAgent, EmailConfig
from .fusion_agent import FusionAgent
from .ocr_agent import OCRAgent


__all__ = [
    "AudioAgent",
    "AudioConfig",
    "BaseAgent",
    "BehavioralAgent",
    "EmailAgent",
    "EmailConfig",
    "FusionAgent",
    "OCRAgent",
]
