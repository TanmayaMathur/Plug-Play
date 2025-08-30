"""
LLM Agent Factory for Voice Agent Framework.
"""

from typing import Dict, Any, Optional
from .base import BaseAgent
from .basic_agent import BasicAgent
from .healthcare_agent import HealthcareAgent
from .customer_support_agent import CustomerSupportAgent
from .custom_llm_agent import CustomLLMAgent
from ..config import LLMConfig
from loguru import logger


class LLMAgentFactory:
    """Factory for creating LLM agents based on configuration."""
    
    @staticmethod
    def create_agent(config: LLMConfig, agent_type: str = "basic") -> BaseAgent:
        """
        Create an LLM agent based on configuration.
        
        Args:
            config: LLM configuration
            agent_type: Type of agent (basic, healthcare, customer_support, custom)
            
        Returns:
            Configured LLM agent
        """
        try:
            if agent_type == "custom":
                return LLMAgentFactory._create_custom_agent(config)
            else:
                return LLMAgentFactory._create_builtin_agent(config, agent_type)
        except Exception as e:
            logger.error(f"Failed to create agent: {e}")
            raise
    
    @staticmethod
    def _create_custom_agent(config: LLMConfig) -> CustomLLMAgent:
        """Create a custom LLM agent."""
        if config.provider != "custom":
            raise ValueError("Custom agent requires provider to be 'custom'")
        
        if not config.api_url:
            raise ValueError("Custom agent requires API URL")
        
        return CustomLLMAgent(
            api_url=config.api_url,
            api_key=config.api_key,
            model=config.model,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            custom_headers=config.custom_headers,
            payload_format=config.custom_payload_format
        )
    
    @staticmethod
    def _create_builtin_agent(config: LLMConfig, agent_type: str) -> BaseAgent:
        """Create a built-in agent with the specified LLM provider."""
        # Get API configuration
        api_config = config.get_api_config()
        
        # Create agent based on type
        if agent_type == "basic":
            return BasicAgent(
                provider=config.provider,
                api_key=api_config["api_key"],
                model=api_config["model"],
                max_tokens=api_config["max_tokens"],
                temperature=api_config["temperature"]
            )
        elif agent_type == "healthcare":
            return HealthcareAgent(
                provider=config.provider,
                api_key=api_config["api_key"],
                model=api_config["model"],
                max_tokens=api_config["max_tokens"],
                temperature=api_config["temperature"]
            )
        elif agent_type == "customer_support":
            return CustomerSupportAgent(
                provider=config.provider,
                api_key=api_config["api_key"],
                model=api_config["model"],
                max_tokens=api_config["max_tokens"],
                temperature=api_config["temperature"]
            )
        else:
            raise ValueError(f"Unsupported agent type: {agent_type}")
    
    @staticmethod
    def get_available_agents() -> Dict[str, str]:
        """Get list of available agent types."""
        return {
            "basic": "General purpose Q&A agent",
            "healthcare": "Medical intake and healthcare assistant",
            "customer_support": "Customer service and support agent",
            "custom": "Custom LLM integration"
        }
    
    @staticmethod
    def get_available_providers() -> Dict[str, str]:
        """Get list of available LLM providers."""
        return {
            "openai": "OpenAI GPT models (GPT-3.5, GPT-4)",
            "anthropic": "Anthropic Claude models",
            "custom": "Custom API endpoint"
        }
    
    @staticmethod
    def validate_config(config: LLMConfig) -> bool:
        """Validate LLM configuration."""
        try:
            if config.provider not in ["openai", "anthropic", "custom"]:
                raise ValueError(f"Unsupported provider: {config.provider}")
            
            if config.provider == "custom":
                if not config.api_url:
                    raise ValueError("Custom provider requires API URL")
            else:
                if not config.api_key:
                    raise ValueError(f"{config.provider} provider requires API key")
            
            if config.max_tokens <= 0:
                raise ValueError("max_tokens must be positive")
            
            if not 0 <= config.temperature <= 2:
                raise ValueError("temperature must be between 0 and 2")
            
            return True
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False
