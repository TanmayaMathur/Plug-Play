"""
Healthcare agent for medical intake and healthcare assistance.
"""

import re
from typing import Optional, Dict, Any, List
from .base import BaseAgent, AgentContext
from loguru import logger
from openai import OpenAI
import anthropic
import os


class HealthcareAgent(BaseAgent):
    """Healthcare agent for medical intake and healthcare assistance."""
    
    def __init__(self, provider: str = "openai", api_key: str = None, 
                 model: str = "gpt-3.5-turbo", max_tokens: int = 500, 
                 temperature: float = 0.7):
        """
        Initialize healthcare agent.
        
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
        
        # Healthcare-specific data
        self.patient_info = {}
        self.emergency_keywords = [
            "chest pain", "heart attack", "stroke", "unconscious", "bleeding",
            "difficulty breathing", "severe pain", "emergency", "911", "ambulance"
        ]
        
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
        Process healthcare-related input.
        
        Args:
            text: User input text
            context: Optional context information
            
        Returns:
            Healthcare response
        """
        try:
            # Check for emergency keywords
            if self._is_emergency(text):
                return self._handle_emergency(text)
            
            # Preprocess input
            processed_text = self.preprocess(text, context)
            
            # Extract patient information
            self._extract_patient_info(processed_text)
            
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
            logger.error(f"Error processing healthcare request: {e}")
            return f"I'm sorry, I encountered an error: {str(e)}"
    
    def _is_emergency(self, text: str) -> bool:
        """Check if the input contains emergency keywords."""
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.emergency_keywords)
    
    def _handle_emergency(self, text: str) -> str:
        """Handle emergency situations."""
        emergency_response = """
🚨 EMERGENCY DETECTED 🚨

If you're experiencing a medical emergency:
1. Call 911 immediately
2. Don't wait for online assistance
3. Seek immediate medical attention

This AI assistant cannot provide emergency medical care.
Please contact emergency services right away.
        """
        logger.warning(f"Emergency detected in input: {text}")
        return emergency_response.strip()
    
    def _extract_patient_info(self, text: str):
        """Extract patient information from text."""
        # Extract age
        age_match = re.search(r'(\d+)\s*(?:years?\s*old|yo)', text.lower())
        if age_match:
            self.patient_info['age'] = int(age_match.group(1))
        
        # Extract symptoms
        symptom_keywords = ['pain', 'fever', 'headache', 'nausea', 'dizziness', 'cough']
        symptoms = []
        for keyword in symptom_keywords:
            if keyword in text.lower():
                symptoms.append(keyword)
        
        if symptoms:
            self.patient_info['symptoms'] = symptoms
    
    def _call_openai(self, text: str) -> str:
        """Call OpenAI API using the new v1.0+ format."""
        try:
            system_prompt = """You are a healthcare assistant. You can help with:
- General health information
- Symptom assessment (non-emergency)
- Medical appointment guidance
- Health and wellness tips

IMPORTANT: You cannot provide medical diagnosis or treatment.
Always recommend consulting a healthcare professional for medical concerns.
            """
            
            messages = [
                {"role": "system", "content": system_prompt},
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
            system_prompt = """You are a healthcare assistant. You can help with:
- General health information
- Symptom assessment (non-emergency)
- Medical appointment guidance
- Health and wellness tips

IMPORTANT: You cannot provide medical diagnosis or treatment.
Always recommend consulting a healthcare professional for medical concerns.
            """
            
            # Build conversation history for Anthropic
            messages = [{"role": "system", "content": system_prompt}]
            for msg in self.conversation_history:
                if msg["role"] in ["user", "assistant"]:
                    messages.append({"role": msg["role"], "content": msg["content"]})
            
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
        """Preprocess healthcare input."""
        # Basic preprocessing
        text = text.strip()
        
        # Remove common speech artifacts
        text = re.sub(r'\b(um|uh|ah|er)\b', '', text, flags=re.IGNORECASE)
        text = ' '.join(text.split())  # Normalize whitespace
        
        return text
    
    def postprocess(self, response: str, context: Optional[AgentContext] = None) -> str:
        """Postprocess healthcare response."""
        if not response or response.strip() == "":
            return "I'm sorry, I couldn't generate a response. Please try again."
        
        # Add medical disclaimer if not present
        if "consult a healthcare professional" not in response.lower():
            response += "\n\nNote: This is general information only. Please consult a healthcare professional for medical advice."
        
        return response.strip()
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get healthcare agent information."""
        return {
            "name": "Healthcare Agent",
            "description": "Medical intake and healthcare assistant",
            "provider": self.provider,
            "model": self.model,
            "patient_info": self.patient_info,
            "conversation_length": len(self.conversation_history)
        }
    
    def reset_conversation(self):
        """Reset conversation and patient information."""
        super().reset_conversation()
        self.patient_info = {}
        logger.info("Healthcare conversation and patient info reset")
