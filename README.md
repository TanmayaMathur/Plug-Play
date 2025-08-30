# Plug & Play Voice Agent Framework

A modular, scalable framework for building voice-first AI applications with a simple pipeline: **Speech-to-Text → LLM Processing → Text-to-Speech**.

## 🚀 Features

- **Modular Architecture**: Easy to swap STT, LLM, and TTS components
- **Real-time Processing**: Low latency conversational experience (<2s)
- **Pluggable LLM Agents**: Bring your own custom logic
- **Multiple Domain Examples**: Healthcare, Customer Support, and more
- **Simple API/SDK**: Easy integration for developers
- **Security First**: No sensitive data in plaintext logs

## 🏗️ Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Speech    │───▶│     STT     │───▶│     LLM     │
│   Input     │    │  (Whisper/  │    │  (Custom    │
│             │    │   Vosk)     │    │   Agent)    │
└─────────────┘    └─────────────┘    └─────────────┘
                                              │
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Speech    │◀───│     TTS     │◀───│   Response  │
│   Output    │    │ (ElevenLabs/│    │   Text      │
│             │    │  pyttsx3)   │    │             │
└─────────────┘    └─────────────┘    └─────────────┘
```

## 📦 Installation

```bash
# Clone the repository
git clone <repository-url>
cd tequity

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

## 🎯 Quick Start

### 1. Basic Usage

```python
from voice_agent import VoiceAgent
from agents.basic_agent import BasicAgent

# Create a voice agent with basic Q&A capabilities
agent = VoiceAgent(
    stt_engine="whisper",
    llm_agent=BasicAgent(),
    tts_engine="elevenlabs"
)

# Start real-time conversation
agent.start_conversation()
```

### 2. Custom Agent Integration

```python
from voice_agent import VoiceAgent
from agents.base import BaseAgent

class HealthcareAgent(BaseAgent):
    def process(self, text: str) -> str:
        # Your custom healthcare logic here
        return f"Healthcare response to: {text}"

# Use your custom agent
agent = VoiceAgent(
    stt_engine="whisper",
    llm_agent=HealthcareAgent(),
    tts_engine="elevenlabs"
)
```

## 🔧 Configuration

### Environment Variables (.env)

```env
# OpenAI (for Whisper STT and GPT LLM)
OPENAI_API_KEY=your_openai_key

# ElevenLabs (for TTS)
ELEVENLABS_API_KEY=your_elevenlabs_key

# Anthropic (alternative LLM)
ANTHROPIC_API_KEY=your_anthropic_key

# Vosk (offline STT)
VOSK_MODEL_PATH=path/to/vosk/model
```

## 📚 API Documentation

### VoiceAgent Class

```python
class VoiceAgent:
    def __init__(
        self,
        stt_engine: str = "whisper",
        llm_agent: BaseAgent = None,
        tts_engine: str = "elevenlabs",
        config: dict = None
    )
    
    def start_conversation(self) -> None
    def process_audio(self, audio_data: bytes) -> bytes
    def stop_conversation(self) -> None
```

### BaseAgent Interface

```python
class BaseAgent(ABC):
    @abstractmethod
    def process(self, text: str) -> str:
        """Process input text and return response"""
        pass
    
    def preprocess(self, text: str) -> str:
        """Optional preprocessing"""
        return text
    
    def postprocess(self, text: str) -> str:
        """Optional postprocessing"""
        return text
```

## 🎭 Domain Examples

### 1. Healthcare Intake Bot

```python
from agents.healthcare_agent import HealthcareAgent

agent = VoiceAgent(
    llm_agent=HealthcareAgent(),
    config={"domain": "healthcare"}
)
```

### 2. Customer Support Agent

```python
from agents.customer_support_agent import CustomerSupportAgent

agent = VoiceAgent(
    llm_agent=CustomerSupportAgent(),
    config={"domain": "customer_support"}
)
```

## 🚀 Running the Demo

```bash
# Start the web interface
python -m voice_agent.web

# Run with specific agent
python -m voice_agent.cli --agent healthcare

# Run tests
pytest tests/
```

## 📊 Performance Metrics

- **Latency**: <2 seconds for small exchanges
- **STT Accuracy**: >95% with Whisper
- **TTS Quality**: Natural-sounding speech
- **Modularity**: <5 lines to swap components

## 🔒 Security Features

- No sensitive data in logs
- Secure API key management
- Audio data encryption in transit
- Configurable data retention

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add your custom agent
4. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details

## 🆘 Support

- Documentation: `/docs`
- Examples: `/examples`
- Issues: GitHub Issues
- Discussions: GitHub Discussions
