"""
Plug & Play Voice Agent Framework

A modular framework for building voice-first AI applications.
"""

from .core import VoiceAgent
from .agents.base import BaseAgent
from .config import VoiceAgentConfig

__version__ = "1.0.0"
__author__ = "Voice Agent Team"

__all__ = ["VoiceAgent", "BaseAgent", "VoiceAgentConfig"]
