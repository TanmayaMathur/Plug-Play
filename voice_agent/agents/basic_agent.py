"""
Basic LLM agent for general Q&A.
"""

import os
from typing import Optional, List, Dict, Any
from .base import BaseAgent, AgentContext
from loguru import logger
from openai import OpenAI
import anthropic


class BasicAgent(BaseAgent):
    """Basic agent for general Q&A using various LLM providers."""
    
    def __init__(self, provider: str = "openai", api_key: str = None, 
                 model: str = "gpt-3.5-turbo", max_tokens: int = 500, 
                 temperature: float = 0.7):
        """
        Initialize basic agent.
        
        Args:
            provider: LLM provider (openai, anthropic)
            api_key: API key for the provider
            model: Model name to use
            max_tokens: Maximum tokens for response
            temperature: Temperature for response generation
        """
        super().__init__()
        self.provider = provider
        self.api_key = api_key or self._get_default_api_key()
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.conversation_history = []
        
        # Initialize client based on provider
        self._initialize_client()
    
    def _get_default_api_key(self) -> str:
        """Get default API key based on provider."""
        if self.provider == "openai":
            return os.getenv("OPENAI_API_KEY", "")
        elif self.provider == "anthropic":
            return os.getenv("ANTHROPIC_API_KEY", "")
        else:
            return ""
    
    def _initialize_client(self):
        """Initialize the appropriate client based on provider."""
        if not self.api_key:
            logger.warning(f"No API key provided for {self.provider}")
            return
        
        if self.provider == "openai":
            self.openai_client = OpenAI(api_key=self.api_key)
        elif self.provider == "anthropic":
            self.anthropic_client = anthropic.Anthropic(api_key=self.api_key)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def process(self, text: str, context: Optional[AgentContext] = None) -> str:
        """
        Process user input and generate response.
        
        Args:
            text: User input text
            context: Optional context information
            
        Returns:
            Generated response
        """
        try:
            # Preprocess input
            processed_text = self.preprocess(text, context)
            
            # Generate response based on provider
            if self.provider == "openai":
                response = self._call_openai(processed_text)
            elif self.provider == "anthropic":
                response = self._call_anthropic(processed_text)
            else:
                response = f"Unsupported provider: {self.provider}"
            
            # Postprocess response
            final_response = self.postprocess(response, context)
            
            # Add to conversation history
            self.add_to_history(text, final_response)
            
            return final_response
            
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            return f"I'm sorry, I encountered an error: {str(e)}"
    
    def _call_openai(self, text: str) -> str:
        """Call OpenAI API using the new v1.0+ format."""
        try:
            messages = [
                {"role": "system", "content": "You are a helpful AI assistant."},
                *self.conversation_history,
                {"role": "user", "content": text}
            ]
            
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise
    
    def _call_anthropic(self, text: str) -> str:
        """Call Anthropic API."""
        try:
            # Build conversation history for Anthropic
            messages = []
            for msg in self.conversation_history:
                if msg["role"] == "user":
                    messages.append({"role": "user", "content": msg["content"]})
                elif msg["role"] == "assistant":
                    messages.append({"role": "assistant", "content": msg["content"]})
            
            # Add current user message
            messages.append({"role": "user", "content": text})
            
            response = self.anthropic_client.messages.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            return response.content[0].text
            
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise
    
    def preprocess(self, text: str, context: Optional[AgentContext] = None) -> str:
        """Preprocess user input."""
        # Basic preprocessing - trim whitespace
        return text.strip()
    
    def postprocess(self, response: str, context: Optional[AgentContext] = None) -> str:
        """Postprocess LLM response."""
        # Basic postprocessing - ensure response is not empty
        if not response or response.strip() == "":
            return "I'm sorry, I couldn't generate a response. Please try again."
        return response.strip()
    
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
            "name": "Basic Agent",
            "description": "General purpose Q&A agent",
            "provider": self.provider,
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "conversation_length": len(self.conversation_history)
        }
    
    def validate_input(self, text: str) -> bool:
        """Validate user input."""
        return text is not None and len(text.strip()) > 0
    
    def handle_error(self, error: Exception) -> str:
        """Handle errors gracefully."""
        logger.error(f"Agent error: {error}")
        return "I'm sorry, I encountered an error. Please try again."
