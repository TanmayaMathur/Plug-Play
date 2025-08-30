"""
Configuration management for Voice Agent Framework.
"""

import os
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class LLMConfig(BaseModel):
    """Configuration for LLM providers."""
    
    # Built-in LLM options
    provider: str = Field(default="openai", description="LLM provider: openai, anthropic, custom")
    
    # API credentials
    api_key: Optional[str] = Field(default=None, description="API key for the LLM provider")
    api_url: Optional[str] = Field(default=None, description="Custom API endpoint URL")
    
    # Model settings
    model: str = Field(default="gpt-3.5-turbo", description="Model name to use")
    max_tokens: int = Field(default=500, description="Maximum tokens for response")
    temperature: float = Field(default=0.7, description="Temperature for response generation")
    
    # Custom LLM settings
    custom_headers: Optional[Dict[str, str]] = Field(default=None, description="Custom headers for API requests")
    custom_payload_format: Optional[str] = Field(default=None, description="Custom payload format: openai, anthropic, custom")
    
    @classmethod
    def from_env(cls) -> "LLMConfig":
        """Create LLM config from environment variables."""
        return cls(
            provider=os.getenv("LLM_PROVIDER", "openai"),
            api_key=os.getenv("LLM_API_KEY"),
            api_url=os.getenv("LLM_API_URL"),
            model=os.getenv("LLM_MODEL", "gpt-3.5-turbo"),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "500")),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.7"))
        )
    
    def get_api_config(self) -> Dict[str, Any]:
        """Get API configuration for the selected provider."""
        if self.provider == "openai":
            return {
                "api_key": self.api_key or os.getenv("OPENAI_API_KEY"),
                "model": self.model,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature
            }
        elif self.provider == "anthropic":
            return {
                "api_key": self.api_key or os.getenv("ANTHROPIC_API_KEY"),
                "model": self.model,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature
            }
        elif self.provider == "custom":
            return {
                "api_key": self.api_key,
                "api_url": self.api_url,
                "model": self.model,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "headers": self.custom_headers,
                "payload_format": self.custom_payload_format
            }
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")


class STTConfig(BaseModel):
    """Configuration for Speech-to-Text engines."""
    
    engine: str = Field(default="whisper", description="STT engine: whisper, vosk")
    api_key: Optional[str] = Field(default=None, description="API key for STT service")
    model_path: Optional[str] = Field(default=None, description="Path to offline model")
    
    @classmethod
    def from_env(cls) -> "STTConfig":
        """Create STT config from environment variables."""
        return cls(
            engine=os.getenv("STT_ENGINE", "whisper"),
            api_key=os.getenv("STT_API_KEY") or os.getenv("OPENAI_API_KEY"),
            model_path=os.getenv("VOSK_MODEL_PATH", "models/vosk-model-small-en-us-0.15")
        )


class TTSConfig(BaseModel):
    """Configuration for Text-to-Speech engines."""
    
    engine: str = Field(default="pyttsx3", description="TTS engine: elevenlabs, pyttsx3")
    api_key: Optional[str] = Field(default=None, description="API key for TTS service")
    voice_id: Optional[str] = Field(default=None, description="Voice ID for TTS service")
    sample_rate: int = Field(default=22050, description="Audio sample rate")
    
    @classmethod
    def from_env(cls) -> "TTSConfig":
        """Create TTS config from environment variables."""
        return cls(
            engine=os.getenv("TTS_ENGINE", "pyttsx3"),
            api_key=os.getenv("TTS_API_KEY") or os.getenv("ELEVENLABS_API_KEY"),
            voice_id=os.getenv("TTS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM"),
            sample_rate=int(os.getenv("TTS_SAMPLE_RATE", "22050"))
        )


class VoiceAgentConfig(BaseModel):
    """Main configuration for Voice Agent Framework."""
    
    # Component configurations
    llm: LLMConfig = Field(default_factory=LLMConfig.from_env, description="LLM configuration")
    stt: STTConfig = Field(default_factory=STTConfig.from_env, description="STT configuration")
    tts: TTSConfig = Field(default_factory=TTSConfig.from_env, description="TTS configuration")
    
    # General settings
    session_timeout: int = Field(default=300, description="Session timeout in seconds")
    max_conversation_history: int = Field(default=20, description="Maximum conversation history items")
    enable_logging: bool = Field(default=True, description="Enable detailed logging")
    log_level: str = Field(default="INFO", description="Logging level")
    
    # Performance settings
    audio_chunk_size: int = Field(default=1024, description="Audio chunk size for processing")
    max_audio_duration: int = Field(default=30, description="Maximum audio duration in seconds")
    
    @classmethod
    def from_env(cls) -> "VoiceAgentConfig":
        """Create configuration from environment variables."""
        return cls(
            llm=LLMConfig.from_env(),
            stt=STTConfig.from_env(),
            tts=TTSConfig.from_env(),
            session_timeout=int(os.getenv("SESSION_TIMEOUT", "300")),
            max_conversation_history=int(os.getenv("MAX_CONVERSATION_HISTORY", "20")),
            enable_logging=os.getenv("ENABLE_LOGGING", "true").lower() == "true",
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            audio_chunk_size=int(os.getenv("AUDIO_CHUNK_SIZE", "1024")),
            max_audio_duration=int(os.getenv("MAX_AUDIO_DURATION", "30"))
        )
    
    def update_llm_config(self, provider: str, api_key: str = None, api_url: str = None, 
                         model: str = None, **kwargs) -> None:
        """Update LLM configuration dynamically."""
        self.llm.provider = provider
        if api_key is not None:
            self.llm.api_key = api_key
        if api_url is not None:
            self.llm.api_url = api_url
        if model is not None:
            self.llm.model = model
        
        # Update any additional parameters
        for key, value in kwargs.items():
            if hasattr(self.llm, key):
                setattr(self.llm, key, value)
    
    def get_llm_config_dict(self) -> Dict[str, Any]:
        """Get LLM configuration as dictionary for agent creation."""
        return self.llm.get_api_config()
    
    def get_stt_config_dict(self) -> Dict[str, Any]:
        """Get STT configuration as dictionary for engine creation."""
        return {
            "engine": self.stt.engine,
            "api_key": self.stt.api_key,
            "model_path": self.stt.model_path
        }
    
    def get_tts_config_dict(self) -> Dict[str, Any]:
        """Get TTS configuration as dictionary for engine creation."""
        return {
            "engine": self.tts.engine,
            "api_key": self.tts.api_key,
            "voice_id": self.tts.voice_id,
            "sample_rate": self.tts.sample_rate
        }
