"""
Custom LLM Agent for integrating any LLM API.
"""

import requests
import json
from typing import Optional, Dict, Any, List
from .base import BaseAgent, AgentContext
from loguru import logger


class CustomLLMAgent(BaseAgent):
    """
    Custom LLM agent that can integrate with any LLM API endpoint.
    
    Supports various API formats including OpenAI-compatible, Anthropic-compatible,
    and custom formats.
    """
    
    def __init__(self, api_url: str, api_key: str = None, model: str = "default",
                 max_tokens: int = 500, temperature: float = 0.7,
                 custom_headers: Dict[str, str] = None, payload_format: str = "openai"):
        """
        Initialize custom LLM agent.
        
        Args:
            api_url: Your LLM API endpoint URL
            api_key: API key for authentication (optional)
            model: Model name to use
            max_tokens: Maximum tokens for response
            temperature: Temperature for response generation
            custom_headers: Custom headers for API requests
            payload_format: API format (openai, anthropic, custom)
        """
        super().__init__()
        self.api_url = api_url
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.custom_headers = custom_headers or {}
        self.payload_format = payload_format
        self.conversation_history = []
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self):
        """Validate agent configuration."""
        if not self.api_url:
            raise ValueError("API URL is required")
        
        if self.payload_format not in ["openai", "anthropic", "custom"]:
            raise ValueError(f"Unsupported payload format: {self.payload_format}")
        
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        
        if not 0 <= self.temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")
    
    def process(self, text: str, context: Optional[AgentContext] = None) -> str:
        """
        Process user input using custom LLM API.
        
        Args:
            text: User input text
            context: Optional context information
            
        Returns:
            LLM response text
        """
        try:
            # Preprocess input
            processed_text = self.preprocess(text, context)
            
            # Generate response
            response = self._call_api(processed_text)
            
            # Postprocess response
            final_response = self.postprocess(response, context)
            
            # Add to conversation history
            self.add_to_history(text, final_response)
            
            return final_response
            
        except Exception as e:
            logger.error(f"Custom LLM API error: {e}")
            return f"I'm sorry, I encountered an error: {str(e)}"
    
    def _call_api(self, text: str) -> str:
        """Make API call to custom LLM endpoint."""
        try:
            # Prepare headers
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            headers.update(self.custom_headers)
            
            # Prepare payload based on format
            payload = self._build_payload(text)
            
            # Make API request
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                return self._parse_response(response.json())
            else:
                raise Exception(f"API request failed with status {response.status_code}: {response.text}")
                
        except requests.exceptions.RequestException as e:
            raise Exception(f"Network error: {e}")
        except json.JSONDecodeError as e:
            raise Exception(f"Invalid JSON response: {e}")
        except Exception as e:
            raise Exception(f"API call failed: {e}")
    
    def _build_payload(self, text: str) -> Dict[str, Any]:
        """Build API payload based on format."""
        if self.payload_format == "openai":
            return self._build_openai_payload(text)
        elif self.payload_format == "anthropic":
            return self._build_anthropic_payload(text)
        elif self.payload_format == "custom":
            return self._build_custom_payload(text)
        else:
            raise ValueError(f"Unsupported payload format: {self.payload_format}")
    
    def _build_openai_payload(self, text: str) -> Dict[str, Any]:
        """Build OpenAI-compatible payload."""
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            *self.conversation_history,
            {"role": "user", "content": text}
        ]
        
        return {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }
    
    def _build_anthropic_payload(self, text: str) -> Dict[str, Any]:
        """Build Anthropic-compatible payload."""
        messages = []
        for msg in self.conversation_history:
            if msg["role"] in ["user", "assistant"]:
                messages.append({"role": msg["role"], "content": msg["content"]})
        
        messages.append({"role": "user", "content": text})
        
        return {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }
    
    def _build_custom_payload(self, text: str) -> Dict[str, Any]:
        """Build custom payload format."""
        return {
            "input": text,
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "history": self.conversation_history
        }
    
    def _parse_response(self, response_data: Dict[str, Any]) -> str:
        """Parse API response based on format."""
        try:
            if self.payload_format == "openai":
                return response_data.get("choices", [{}])[0].get("message", {}).get("content", "")
            elif self.payload_format == "anthropic":
                return response_data.get("content", [{}])[0].get("text", "")
            elif self.payload_format == "custom":
                # Try common response formats
                if "response" in response_data:
                    return response_data["response"]
                elif "text" in response_data:
                    return response_data["text"]
                elif "content" in response_data:
                    return response_data["content"]
                elif "message" in response_data:
                    return response_data["message"]
                else:
                    # Return the entire response as string
                    return str(response_data)
            else:
                return str(response_data)
        except (KeyError, IndexError, TypeError) as e:
            logger.warning(f"Failed to parse response: {e}")
            return str(response_data)
    
    def preprocess(self, text: str, context: Optional[AgentContext] = None) -> str:
        """Preprocess user input."""
        return text.strip()
    
    def postprocess(self, response: str, context: Optional[AgentContext] = None) -> str:
        """Postprocess LLM response."""
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
            "name": "Custom LLM Agent",
            "description": "Custom LLM integration",
            "api_url": self.api_url,
            "model": self.model,
            "payload_format": self.payload_format,
            "has_api_key": bool(self.api_key),
            "conversation_length": len(self.conversation_history)
        }
    
    def validate_input(self, text: str) -> bool:
        """Validate user input."""
        return text is not None and len(text.strip()) > 0
    
    def handle_error(self, error: Exception) -> str:
        """Handle errors gracefully."""
        logger.error(f"Custom LLM agent error: {error}")
        return "I'm sorry, I encountered an error. Please try again."
    
    def test_connection(self) -> bool:
        """Test API connection."""
        try:
            # Make a simple test call
            test_response = self._call_api("Hello")
            return bool(test_response and test_response.strip())
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
