"""
Customer support agent for customer service and support.
"""

import re
from typing import Optional, Dict, Any, List
from .base import BaseAgent, AgentContext
from loguru import logger
from openai import OpenAI
import anthropic
import os


class CustomerSupportAgent(BaseAgent):
    """Customer support agent for customer service and support."""
    
    def __init__(self, provider: str = "openai", api_key: str = None, 
                 model: str = "gpt-3.5-turbo", max_tokens: int = 500, 
                 temperature: float = 0.7):
        """
        Initialize customer support agent.
        
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
        
        # Customer support data
        self.customer_info = {}
        self.escalation_keywords = [
            "manager", "supervisor", "escalate", "complaint", "unhappy",
            "dissatisfied", "angry", "frustrated", "terrible service"
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
        Process customer support input.
        
        Args:
            text: User input text
            context: Optional context information
            
        Returns:
            Customer support response
        """
        try:
            # Check for escalation keywords
            if self._needs_escalation(text):
                return self._handle_escalation(text)
            
            # Preprocess input
            processed_text = self.preprocess(text, context)
            
            # Extract customer information
            self._extract_customer_info(processed_text)
            
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
            logger.error(f"Error processing customer support request: {e}")
            return f"I'm sorry, I encountered an error: {str(e)}"
    
    def _needs_escalation(self, text: str) -> bool:
        """Check if the input requires escalation."""
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.escalation_keywords)
    
    def _handle_escalation(self, text: str) -> str:
        """Handle escalation requests."""
        escalation_response = """
🔄 ESCALATION REQUESTED 🔄

I understand you'd like to speak with a supervisor or manager.
I'll connect you with a senior customer service representative.

Please hold while I transfer you to someone who can better assist you.
        """
        logger.info(f"Escalation requested in input: {text}")
        return escalation_response.strip()
    
    def _extract_customer_info(self, text: str):
        """Extract customer information from text."""
        # Extract account number
        account_match = re.search(r'account\s*(?:number|#)?\s*(\d+)', text.lower())
        if account_match:
            self.customer_info['account_number'] = account_match.group(1)
        
        # Extract order number
        order_match = re.search(r'order\s*(?:number|#)?\s*(\d+)', text.lower())
        if order_match:
            self.customer_info['order_number'] = order_match.group(1)
        
        # Extract issue type
        issue_keywords = ['refund', 'return', 'shipping', 'billing', 'technical', 'login']
        issues = []
        for keyword in issue_keywords:
            if keyword in text.lower():
                issues.append(keyword)
        
        if issues:
            self.customer_info['issue_type'] = issues
    
    def _call_openai(self, text: str) -> str:
        """Call OpenAI API using the new v1.0+ format."""
        try:
            system_prompt = """You are a customer service representative. You can help with:
- General customer inquiries
- Order status and tracking
- Product information
- Basic troubleshooting
- Return and refund policies

Be helpful, professional, and empathetic. Always try to resolve issues when possible.
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
            system_prompt = """You are a customer service representative. You can help with:
- General customer inquiries
- Order status and tracking
- Product information
- Basic troubleshooting
- Return and refund policies

Be helpful, professional, and empathetic. Always try to resolve issues when possible.
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
        """Preprocess customer support input."""
        # Basic preprocessing
        text = text.strip()
        
        # Remove common speech artifacts
        text = re.sub(r'\b(um|uh|ah|er)\b', '', text, flags=re.IGNORECASE)
        text = ' '.join(text.split())  # Normalize whitespace
        
        return text
    
    def postprocess(self, response: str, context: Optional[AgentContext] = None) -> str:
        """Postprocess customer support response."""
        if not response or response.strip() == "":
            return "I'm sorry, I couldn't generate a response. Please try again."
        
        # Add customer service closing if appropriate
        if "thank you" not in response.lower() and "anything else" not in response.lower():
            response += "\n\nIs there anything else I can help you with today?"
        
        return response.strip()
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get customer support agent information."""
        return {
            "name": "Customer Support Agent",
            "description": "Customer service and support agent",
            "provider": self.provider,
            "model": self.model,
            "customer_info": self.customer_info,
            "conversation_length": len(self.conversation_history)
        }
    
    def reset_conversation(self):
        """Reset conversation and customer information."""
        super().reset_conversation()
        self.customer_info = {}
        logger.info("Customer support conversation and customer info reset")
