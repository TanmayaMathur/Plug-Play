"""
Core Voice Agent implementation.
"""

import asyncio
import time
from typing import Optional, Dict, Any, Callable
from .config import VoiceAgentConfig, LLMConfig
from .agents.factory import LLMAgentFactory
from .engines.stt_engine import create_stt_engine
from .engines.tts_engine import create_tts_engine
from .agents.base import BaseAgent, AgentContext
from loguru import logger


class VoiceAgent:
    """
    Main Voice Agent class that orchestrates STT → LLM → TTS pipeline.
    
    This class manages the entire voice interaction flow, including:
    - Speech-to-Text conversion
    - LLM processing
    - Text-to-Speech synthesis
    - Session management
    - Performance tracking
    """
    
    def __init__(self, config: Optional[VoiceAgentConfig] = None, 
                 llm_agent: Optional[BaseAgent] = None,
                 stt_engine: Optional[str] = None,
                 tts_engine: Optional[str] = None):
        """
        Initialize Voice Agent.
        
        Args:
            config: Voice Agent configuration
            llm_agent: Pre-configured LLM agent (optional)
            stt_engine: STT engine type (optional, overrides config)
            tts_engine: TTS engine type (optional, overrides config)
        """
        # Load configuration
        self.config = config or VoiceAgentConfig.from_env()
        
        # Initialize components
        self.llm_agent = llm_agent
        self.stt_engine = None
        self.tts_engine = None
        
        # Session management
        self.session_id = None
        self.conversation_start_time = None
        self.total_interactions = 0
        
        # Performance tracking
        self.performance_metrics = {
            "total_latency": 0.0,
            "stt_latency": 0.0,
            "llm_latency": 0.0,
            "tts_latency": 0.0,
            "interaction_count": 0
        }
        
        # Callbacks
        self.on_interaction_start: Optional[Callable] = None
        self.on_interaction_end: Optional[Callable] = None
        self.on_error: Optional[Callable] = None
        
        # Initialize engines
        self._initialize_engines(stt_engine, tts_engine)
        
        logger.info("Voice Agent initialized successfully")
    
    def _initialize_engines(self, stt_engine: Optional[str] = None, tts_engine: Optional[str] = None):
        """Initialize STT and TTS engines."""
        try:
            # Initialize STT engine
            stt_config = self.config.get_stt_config_dict()
            if stt_engine:
                stt_config["engine"] = stt_engine
            self.stt_engine = create_stt_engine(stt_config["engine"], stt_config)
            
            # Initialize TTS engine
            tts_config = self.config.get_tts_config_dict()
            if tts_engine:
                tts_config["engine"] = tts_engine
            self.tts_engine = create_tts_engine(tts_config["engine"], tts_config)
            
            logger.info(f"Engines initialized: STT={stt_config['engine']}, TTS={tts_config['engine']}")
            
        except Exception as e:
            logger.error(f"Failed to initialize engines: {e}")
            raise
    
    def configure_llm(self, provider: str, api_key: str = None, api_url: str = None,
                     model: str = None, agent_type: str = "basic", **kwargs):
        """
        Configure LLM dynamically.
        
        Args:
            provider: LLM provider (openai, anthropic, custom)
            api_key: API key for the provider
            api_url: Custom API URL (for custom provider)
            model: Model name to use
            agent_type: Type of agent (basic, healthcare, customer_support, custom)
            **kwargs: Additional configuration parameters
        """
        try:
            # Update configuration
            self.config.update_llm_config(provider, api_key, api_url, model, **kwargs)
            
            # Create new agent
            llm_config = self.config.llm
            self.llm_agent = LLMAgentFactory.create_agent(llm_config, agent_type)
            
            logger.info(f"LLM configured: provider={provider}, agent_type={agent_type}")
            
        except Exception as e:
            logger.error(f"Failed to configure LLM: {e}")
            raise
    
    def start_session(self, session_id: Optional[str] = None):
        """Start a new conversation session."""
        self.session_id = session_id or f"session_{int(time.time())}"
        self.conversation_start_time = time.time()
        self.total_interactions = 0
        
        if self.llm_agent:
            self.llm_agent.reset_conversation()
        
        logger.info(f"Session started: {self.session_id}")
    
    def end_session(self):
        """End the current conversation session."""
        if self.session_id:
            session_duration = time.time() - self.conversation_start_time
            logger.info(f"Session ended: {self.session_id}, duration: {session_duration:.2f}s")
            
            self.session_id = None
            self.conversation_start_time = None
    
    async def process_audio(self, audio_data: bytes) -> bytes:
        """
        Process audio input and return audio response.
        
        Args:
            audio_data: Raw audio data
            
        Returns:
            Audio response data
        """
        start_time = time.time()
        
        try:
            # Call interaction start callback
            if self.on_interaction_start:
                self.on_interaction_start()
            
            # Step 1: Speech-to-Text
            stt_start = time.time()
            text = await self._speech_to_text(audio_data)
            self.performance_metrics["stt_latency"] += time.time() - stt_start
            
            if not text or not text.strip():
                return b""  # No audio response for empty input
            
            # Step 2: LLM Processing
            llm_start = time.time()
            response_text = await self._process_with_llm(text)
            self.performance_metrics["llm_latency"] += time.time() - llm_start
            
            if not response_text:
                return b""  # No audio response for empty response
            
            # Step 3: Text-to-Speech
            tts_start = time.time()
            audio_response = await self._text_to_speech(response_text)
            self.performance_metrics["tts_latency"] += time.time() - tts_start
            
            # Update metrics
            total_latency = time.time() - start_time
            self.performance_metrics["total_latency"] += total_latency
            self.performance_metrics["interaction_count"] += 1
            self.total_interactions += 1
            
            # Call interaction end callback
            if self.on_interaction_end:
                self.on_interaction_end(text, response_text, total_latency)
            
            logger.info(f"Interaction completed in {total_latency:.2f}s")
            return audio_response
            
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            
            # Call error callback
            if self.on_error:
                self.on_error(e)
            
            return b""
    
    async def _speech_to_text(self, audio_data: bytes) -> str:
        """Convert speech to text."""
        try:
            if not self.stt_engine:
                raise ValueError("STT engine not initialized")
            
            text = self.stt_engine.transcribe(audio_data)
            logger.info(f"STT result: {text[:50]}...")
            return text
            
        except Exception as e:
            logger.error(f"STT error: {e}")
            raise
    
    async def _process_with_llm(self, text: str) -> str:
        """Process text with LLM agent."""
        try:
            if not self.llm_agent:
                raise ValueError("LLM agent not initialized")
            
            # Create context
            context = AgentContext(
                session_id=self.session_id,
                metadata={
                    "timestamp": time.time(),
                    "interaction_count": self.total_interactions,
                    "session_duration": time.time() - self.conversation_start_time if self.conversation_start_time else 0
                }
            )
            
            response = self.llm_agent.process(text, context)
            logger.info(f"LLM response: {response[:50]}...")
            return response
            
        except Exception as e:
            logger.error(f"LLM error: {e}")
            raise
    
    async def _text_to_speech(self, text: str) -> bytes:
        """Convert text to speech."""
        try:
            if not self.tts_engine:
                raise ValueError("TTS engine not initialized")
            
            audio_data = self.tts_engine.synthesize(text)
            logger.info(f"TTS generated {len(audio_data)} bytes")
            return audio_data
            
        except Exception as e:
            logger.error(f"TTS error: {e}")
            raise
    
    def process_text(self, text: str) -> str:
        """
        Process text input directly (without STT/TTS).
        
        Args:
            text: Input text
            
        Returns:
            Response text
        """
        try:
            if not self.llm_agent:
                raise ValueError("LLM agent not initialized")
            
            context = AgentContext(
                session_id=self.session_id,
                metadata={
                    "timestamp": time.time(),
                    "interaction_count": self.total_interactions
                }
            )
            
            response = self.llm_agent.process(text, context)
            self.total_interactions += 1
            
            return response
            
        except Exception as e:
            logger.error(f"Text processing error: {e}")
            return f"Error: {str(e)}"
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        metrics = self.performance_metrics.copy()
        
        if metrics["interaction_count"] > 0:
            metrics["avg_total_latency"] = metrics["total_latency"] / metrics["interaction_count"]
            metrics["avg_stt_latency"] = metrics["stt_latency"] / metrics["interaction_count"]
            metrics["avg_llm_latency"] = metrics["llm_latency"] / metrics["interaction_count"]
            metrics["avg_tts_latency"] = metrics["tts_latency"] / metrics["interaction_count"]
        
        return metrics
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get agent information."""
        if not self.llm_agent:
            return {"error": "No LLM agent configured"}
        
        return self.llm_agent.get_agent_info()
    
    def reset_metrics(self):
        """Reset performance metrics."""
        self.performance_metrics = {
            "total_latency": 0.0,
            "stt_latency": 0.0,
            "llm_latency": 0.0,
            "tts_latency": 0.0,
            "interaction_count": 0
        }
        logger.info("Performance metrics reset")
    
    def start_conversation(self):
        """Start an interactive conversation (for CLI usage)."""
        if not self.llm_agent:
            raise ValueError("LLM agent not configured")
        
        self.start_session()
        print("🎤 Voice Agent Conversation Started")
        print("Type 'quit' to exit, 'reset' to reset conversation")
        print("-" * 50)
        
        try:
            while True:
                user_input = input("You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                elif user_input.lower() == 'reset':
                    self.llm_agent.reset_conversation()
                    print("Conversation reset!")
                    continue
                elif not user_input:
                    continue
                
                response = self.process_text(user_input)
                print(f"Assistant: {response}")
                
        except KeyboardInterrupt:
            print("\nConversation ended by user")
        finally:
            self.end_session()
            print("Session ended")
