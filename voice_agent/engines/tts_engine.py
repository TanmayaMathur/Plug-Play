"""
Text-to-Speech engine implementations.
"""

import os
import tempfile
import io
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import pyttsx3
from elevenlabs import text_to_speech
from loguru import logger


class TTSEngine(ABC):
    """Base class for Text-to-Speech engines."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize TTS engine with configuration.
        
        Args:
            config: Engine configuration
        """
        self.config = config
        self.sample_rate = config.get("sample_rate", 22050)
        self.voice = config.get("voice", "default")
        self.speed = config.get("speed", 1.0)
    
    @abstractmethod
    def synthesize(self, text: str) -> bytes:
        """
        Synthesize text to speech.
        
        Args:
            text: Text to synthesize
            
        Returns:
            Audio data as bytes
        """
        pass
    
    def postprocess_audio(self, audio_data: bytes) -> bytes:
        """
        Postprocess audio data after synthesis.
        
        Args:
            audio_data: Raw audio data
            
        Returns:
            Processed audio data
        """
        return audio_data


class ElevenLabsTTS(TTSEngine):
    """ElevenLabs-based TTS engine."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get("api_key")
        if not self.api_key:
            raise ValueError("ElevenLabs API key is required")
        
        self.client = text_to_speech.TextToSpeechClient(api_key=self.api_key)
        self.voice_id = config.get("voice_id", "21m00Tcm4TlvDq8ikWAM")  # Rachel voice
    
    def synthesize(self, text: str) -> bytes:
        """
        Synthesize text using ElevenLabs.
        
        Args:
            text: Text to synthesize
            
        Returns:
            Audio data as bytes
        """
        try:
            logger.info(f"ElevenLabs synthesizing: {text[:50]}...")
            
            # Generate audio using ElevenLabs
            response = self.client.convert(
                text=text,
                voice_id=self.voice_id,
                output_format="mp3"
            )
            
            # Get audio data directly
            audio_data = response.content
            
            logger.info("ElevenLabs synthesis completed")
            return audio_data
            
        except Exception as e:
            logger.error(f"ElevenLabs synthesis failed: {e}")
            return b""
    
    def postprocess_audio(self, audio_data: bytes) -> bytes:
        """
        Postprocess ElevenLabs audio.
        
        Args:
            audio_data: Raw audio data
            
        Returns:
            Processed audio data
        """
        # ElevenLabs returns MP3 format, which is already optimized
        return audio_data


class Pyttsx3TTS(TTSEngine):
    """Pyttsx3-based TTS engine for offline synthesis."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.engine = None
        self._initialize_engine()
    
    def _initialize_engine(self):
        """Initialize pyttsx3 engine."""
        try:
            logger.info("Initializing pyttsx3 engine")
            self.engine = pyttsx3.init()
            
            # Configure voice
            voices = self.engine.getProperty('voices')
            if voices:
                # Try to find a female voice, otherwise use the first available
                female_voice = None
                for voice in voices:
                    if "female" in voice.name.lower():
                        female_voice = voice.id
                        break
                
                if female_voice:
                    self.engine.setProperty('voice', female_voice)
                else:
                    self.engine.setProperty('voice', voices[0].id)
            
            # Configure speech rate
            self.engine.setProperty('rate', int(200 * self.speed))  # Default 200 WPM
            
            # Configure volume
            self.engine.setProperty('volume', 0.9)
            
            logger.info("Pyttsx3 engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize pyttsx3 engine: {e}")
            raise
    
    def synthesize(self, text: str) -> bytes:
        """
        Synthesize text using pyttsx3.
        
        Args:
            text: Text to synthesize
            
        Returns:
            Audio data as bytes
        """
        try:
            logger.info(f"Pyttsx3 synthesizing: {text[:50]}...")
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file_path = temp_file.name
            
            # Synthesize and save to file
            self.engine.save_to_file(text, temp_file_path)
            self.engine.runAndWait()
            
            # Read audio data
            with open(temp_file_path, "rb") as f:
                audio_data = f.read()
            
            # Clean up temporary file
            os.unlink(temp_file_path)
            
            logger.info("Pyttsx3 synthesis completed")
            return audio_data
            
        except Exception as e:
            logger.error(f"Pyttsx3 synthesis failed: {e}")
            return b""
    
    def postprocess_audio(self, audio_data: bytes) -> bytes:
        """
        Postprocess pyttsx3 audio.
        
        Args:
            audio_data: Raw audio data
            
        Returns:
            Processed audio data
        """
        # Pyttsx3 returns WAV format, which is already suitable
        return audio_data
    
    def get_available_voices(self) -> list:
        """
        Get list of available voices.
        
        Returns:
            List of voice information
        """
        if not self.engine:
            return []
        
        voices = self.engine.getProperty('voices')
        return [
            {
                "id": voice.id,
                "name": voice.name,
                "languages": voice.languages,
                "gender": voice.gender
            }
            for voice in voices
        ]
    
    def set_voice(self, voice_id: str):
        """
        Set the voice to use for synthesis.
        
        Args:
            voice_id: Voice identifier
        """
        if self.engine:
            self.engine.setProperty('voice', voice_id)
    
    def set_speech_rate(self, rate: float):
        """
        Set the speech rate.
        
        Args:
            rate: Speech rate multiplier (1.0 = normal speed)
        """
        if self.engine:
            self.engine.setProperty('rate', int(200 * rate))


def create_tts_engine(engine_type: str, config: Dict[str, Any]) -> TTSEngine:
    """
    Factory function to create TTS engine.
    
    Args:
        engine_type: Type of TTS engine ("elevenlabs" or "pyttsx3")
        config: Engine configuration
        
    Returns:
        TTS engine instance
    """
    if engine_type == "elevenlabs":
        return ElevenLabsTTS(config)
    elif engine_type == "pyttsx3":
        return Pyttsx3TTS(config)
    else:
        raise ValueError(f"Unsupported TTS engine: {engine_type}")
