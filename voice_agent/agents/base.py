"""
Base classes for LLM agents.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from loguru import logger


@dataclass
class AgentContext:
    """Context information for agent processing."""
    
    session_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.metadata is None:
            self.metadata = {}


class BaseAgent(ABC):
    """Base class for all LLM agents."""
    
    def __init__(self):
        """Initialize base agent."""
        self.conversation_history = []
    
    @abstractmethod
    def process(self, text: str, context: Optional[AgentContext] = None) -> str:
        """
        Process input text and generate response.
        
        Args:
            text: Input text from STT
            context: Optional context information
            
        Returns:
            Response text for TTS
        """
        pass
    
    def preprocess(self, text: str, context: Optional[AgentContext] = None) -> str:
        """
        Preprocess input text.
        
        Args:
            text: Raw input text
            context: Optional context information
            
        Returns:
            Preprocessed text
        """
        # Basic preprocessing - trim whitespace
        return text.strip()
    
    def postprocess(self, text: str, context: Optional[AgentContext] = None) -> str:
        """
        Postprocess response text.
        
        Args:
            text: Raw response text
            context: Optional context information
            
        Returns:
            Postprocessed text
        """
        # Basic postprocessing - ensure response is not empty
        if not text or text.strip() == "":
            return "I'm sorry, I couldn't generate a response. Please try again."
        return text.strip()
    
    def add_to_history(self, user_input: str, assistant_response: str):
        """Add conversation to history."""
        self.conversation_history.extend([
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": assistant_response}
        ])
        
        # Keep only last 10 exchanges to manage memory
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]
    
    def get_conversation_summary(self) -> str:
        """Get a summary of the conversation."""
        if not self.conversation_history:
            return "No conversation history."
        
        exchanges = len(self.conversation_history) // 2
        return f"Conversation has {exchanges} exchanges."
    
    def reset_conversation(self):
        """Reset conversation history."""
        self.conversation_history = []
        logger.info("Conversation history reset")
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get agent information."""
        return {
            "name": "Base Agent",
            "description": "Base agent class",
            "conversation_length": len(self.conversation_history)
        }
    
    def validate_input(self, text: str) -> bool:
        """Validate user input."""
        return text is not None and len(text.strip()) > 0
    
    def handle_error(self, error: Exception) -> str:
        """Handle errors gracefully."""
        logger.error(f"Agent error: {error}")
        return "I'm sorry, I encountered an error. Please try again."
