"""
Speech-to-Text engine implementations.
"""

import os
import tempfile
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import whisper
import vosk
from loguru import logger


class STTEngine(ABC):
    """Base class for Speech-to-Text engines."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize STT engine with configuration.
        
        Args:
            config: Engine configuration
        """
        self.config = config
        self.sample_rate = config.get("sample_rate", 16000)
        self.language = config.get("language", "en")
    
    @abstractmethod
    def transcribe(self, audio_data: bytes) -> str:
        """
        Transcribe audio data to text.
        
        Args:
            audio_data: Raw audio data
            
        Returns:
            Transcribed text
        """
        pass
    
    def preprocess_audio(self, audio_data: bytes) -> bytes:
        """
        Preprocess audio data before transcription.
        
        Args:
            audio_data: Raw audio data
            
        Returns:
            Preprocessed audio data
        """
        return audio_data


class WhisperSTT(STTEngine):
    """Whisper-based STT engine."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_name = config.get("model", "base")
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load Whisper model."""
        try:
            logger.info(f"Loading Whisper model: {self.model_name}")
            self.model = whisper.load_model(self.model_name)
            logger.info("Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise
    
    def transcribe(self, audio_data: bytes) -> str:
        """
        Transcribe audio using Whisper.
        
        Args:
            audio_data: Raw audio data
            
        Returns:
            Transcribed text
        """
        try:
            # Save audio data to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_file_path = temp_file.name
            
            print(f"Whisper processing audio file: {temp_file_path}")
            print(f"Audio file size: {len(audio_data)} bytes")
            
            # Try with different Whisper parameters for better speech detection
            result = self.model.transcribe(
                temp_file_path,
                language=self.language,
                fp16=False,  # Use CPU for compatibility
                condition_on_previous_text=False,  # Don't condition on previous text
                temperature=0.0,  # Deterministic output
                compression_ratio_threshold=2.4,  # Lower threshold for better detection
                logprob_threshold=-1.0,  # Lower threshold for better detection
                no_speech_threshold=0.6  # Lower threshold to detect speech
            )
            
            # Extract transcribed text
            transcribed_text = result["text"].strip()
            
            # Check if Whisper detected speech
            if "no_speech_prob" in result:
                no_speech_prob = result["no_speech_prob"]
                print(f"Whisper no_speech_prob: {no_speech_prob}")
                if no_speech_prob > 0.8:
                    print("Whisper thinks this is not speech")
            
            # If no speech detected, try with more aggressive parameters
            if not transcribed_text:
                print("No speech detected, trying with more aggressive parameters...")
                result = self.model.transcribe(
                    temp_file_path,
                    language=self.language,
                    fp16=False,
                    condition_on_previous_text=False,
                    temperature=0.1,  # Slightly more flexible
                    compression_ratio_threshold=3.0,  # Even lower threshold
                    logprob_threshold=-2.0,  # Even lower threshold
                    no_speech_threshold=0.4  # Much lower threshold
                )
                transcribed_text = result["text"].strip()
                print(f"Second attempt result: {transcribed_text}")
            
            # If still no speech, try with extremely aggressive parameters
            if not transcribed_text:
                print("Still no speech, trying with extremely aggressive parameters...")
                result = self.model.transcribe(
                    temp_file_path,
                    language=self.language,
                    fp16=False,
                    condition_on_previous_text=False,
                    temperature=0.2,  # More flexible
                    compression_ratio_threshold=5.0,  # Very low threshold
                    logprob_threshold=-5.0,  # Very low threshold
                    no_speech_threshold=0.2  # Extremely low threshold
                )
                transcribed_text = result["text"].strip()
                print(f"Third attempt result: {transcribed_text}")
            
            # If still no speech, try without any thresholds
            if not transcribed_text:
                print("Final attempt: trying without any thresholds...")
                result = self.model.transcribe(
                    temp_file_path,
                    language=self.language,
                    fp16=False,
                    condition_on_previous_text=False,
                    temperature=0.3,  # More flexible
                    # Remove all thresholds for maximum sensitivity
                )
                transcribed_text = result["text"].strip()
                print(f"Final attempt result: {transcribed_text}")
            
            # Clean up temporary file
            os.unlink(temp_file_path)
            
            logger.info(f"Whisper transcription: {transcribed_text}")
            return transcribed_text
            
        except Exception as e:
            logger.error(f"Whisper transcription failed: {e}")
            print(f"Whisper error details: {e}")
            return ""
    
    def preprocess_audio(self, audio_data: bytes) -> bytes:
        """
        Preprocess audio for Whisper.
        
        Args:
            audio_data: Raw audio data
            
        Returns:
            Preprocessed audio data
        """
        # Whisper can handle various audio formats directly
        return audio_data


class VoskSTT(STTEngine):
    """Vosk-based STT engine for offline transcription."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_path = config.get("model_path", "models/vosk-model-small-en-us-0.15")
        self.model = None
        self.recognizer = None
        self._load_model()
    
    def _load_model(self):
        """Load Vosk model."""
        try:
            logger.info(f"Loading Vosk model from: {self.model_path}")
            
            # Check if model exists
            if not os.path.exists(self.model_path):
                logger.warning(f"Vosk model not found at {self.model_path}")
                logger.info("Please download a Vosk model from https://alphacephei.com/vosk/models")
                raise FileNotFoundError(f"Vosk model not found: {self.model_path}")
            
            self.model = vosk.Model(self.model_path)
            self.recognizer = vosk.KaldiRecognizer(self.model, self.sample_rate)
            logger.info("Vosk model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load Vosk model: {e}")
            raise
    
    def transcribe(self, audio_data: bytes) -> str:
        """
        Transcribe audio using Vosk.
        
        Args:
            audio_data: Raw audio data
            
        Returns:
            Transcribed text
        """
        try:
            # Reset recognizer for new audio
            self.recognizer.Reset()
            
            # Process audio data
            if self.recognizer.AcceptWaveform(audio_data):
                result = self.recognizer.Result()
            else:
                result = self.recognizer.PartialResult()
            
            # Parse JSON result
            import json
            result_dict = json.loads(result)
            
            # Extract transcribed text
            transcribed_text = result_dict.get("text", "").strip()
            
            logger.info(f"Vosk transcription: {transcribed_text}")
            return transcribed_text
            
        except Exception as e:
            logger.error(f"Vosk transcription failed: {e}")
            return ""
    
    def preprocess_audio(self, audio_data: bytes) -> bytes:
        """
        Preprocess audio for Vosk.
        
        Args:
            audio_data: Raw audio data
            
        Returns:
            Preprocessed audio data
        """
        # Vosk expects 16-bit PCM audio
        # Convert if necessary
        return audio_data


def create_stt_engine(engine_type: str, config: Dict[str, Any]) -> STTEngine:
    """
    Factory function to create STT engine.
    
    Args:
        engine_type: Type of STT engine ("whisper" or "vosk")
        config: Engine configuration
        
    Returns:
        STT engine instance
    """
    if engine_type == "whisper":
        return WhisperSTT(config)
    elif engine_type == "vosk":
        return VoskSTT(config)
    else:
        raise ValueError(f"Unsupported STT engine: {engine_type}")
