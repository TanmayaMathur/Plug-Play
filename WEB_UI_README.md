# 🎭 Voice Agent Web UI

A beautiful, industry-ready web interface for your Voice Agent framework with natural speech capabilities.

## ✨ Features

- **🎨 Modern Design**: Clean, professional UI with gradient backgrounds and smooth animations
- **🎤 Voice Recording**: Click-to-record interface with visual feedback
- **🤖 LLM Configuration**: Easy setup for inbuilt or custom LLM providers
- **🗣️ Natural TTS**: Azure Cognitive Services integration for human-like voices
- **📱 Responsive**: Works perfectly on desktop, tablet, and mobile devices
- **💬 Conversation History**: Track all your voice interactions
- **🔧 Real-time Processing**: Live audio processing with progress indicators

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements_web.txt
```

### 2. Run the Web Server

```bash
python app.py
```

### 3. Open Your Browser

Navigate to: `http://localhost:5000`

## 🎯 How to Use

### Step 1: Configure Your Agent
1. **Choose LLM Type**:
   - **🤖 Inbuilt LLM**: Uses OpenRouter (GPT-3.5) - no setup required
   - **🔗 Custom LLM**: Use your own LLM API endpoint

2. **Set Context**: Define your work domain (e.g., "customer support", "healthcare")

3. **Click "Setup Agent"** to initialize

### Step 2: Start Voice Interaction
1. **Click the microphone button** to start recording
2. **Speak your question** clearly
3. **Click again to stop** recording
4. **Wait for processing** (STT → LLM → TTS)
5. **Hear the AI response** in natural voice!

### Step 3: View History
- All conversations are automatically saved
- View transcription and AI responses
- Replay audio responses (when available)

## 🏗️ Architecture

```
Frontend (HTML/CSS/JS) ←→ Flask Backend ←→ Voice Agent Framework
     ↓                           ↓                    ↓
Modern UI                API Endpoints         STT → LLM → TTS
Responsive Design        Audio Processing      Azure TTS
Real-time Updates       Conversation Mgmt     Natural Voices
```

## 🔧 Configuration

### Azure TTS Setup
The web UI uses Azure Cognitive Services for natural TTS voices:

```python
AZURE_TTS_KEY = "your_azure_key_here"
AZURE_TTS_REGION = "your_region"
AZURE_VOICE = "hi-IN-MadhurNeural"  # Natural Hindi voice
```

### LLM Providers
- **Inbuilt**: OpenRouter (GPT-3.5) - configured automatically
- **Custom**: Any OpenAI-compatible API endpoint

## 📱 UI Components

### 1. Configuration Card
- LLM type selection
- API endpoint input (for custom LLM)
- Context/domain setting
- Connection status indicator

### 2. Voice Control Card
- Large recording button
- Visual recording state
- Processing indicators
- Audio playback controls

### 3. Conversation History
- Side-by-side user/AI display
- Timestamp tracking
- Audio replay buttons
- Responsive grid layout

## 🎨 Design Features

- **Gradient Backgrounds**: Modern purple-blue gradients
- **Card-based Layout**: Clean, organized information display
- **Smooth Animations**: Hover effects and transitions
- **Icon Integration**: Font Awesome icons throughout
- **Color-coded Status**: Visual connection indicators
- **Mobile Responsive**: Adapts to any screen size

## 🔒 Security Features

- **API Key Protection**: Secure input fields for sensitive data
- **Session Management**: Flask session handling
- **Input Validation**: Client and server-side validation
- **Error Handling**: Graceful error messages

## 🚀 Production Deployment

### Environment Variables
```bash
export FLASK_ENV=production
export FLASK_SECRET_KEY=your-secret-key
export AZURE_TTS_KEY=your-azure-key
export AZURE_TTS_REGION=your-region
```

### WSGI Server
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Docker (Optional)
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements_web.txt .
RUN pip install -r requirements_web.txt
COPY . .
EXPOSE 5000
CMD ["python", "app.py"]
```

## 🐛 Troubleshooting

### Common Issues

1. **Microphone Access Denied**
   - Check browser permissions
   - Ensure HTTPS in production (required for media access)

2. **Audio Not Playing**
   - Check Azure TTS configuration
   - Verify browser audio support

3. **Recording Not Working**
   - Check microphone permissions
   - Ensure agent is configured

4. **LLM Connection Failed**
   - Verify API keys and endpoints
   - Check network connectivity

### Debug Mode
```bash
export FLASK_DEBUG=1
python app.py
```

## 🔮 Future Enhancements

- [ ] **Real-time Streaming**: Live audio streaming
- [ ] **Voice Cloning**: Custom voice training
- [ ] **Multi-language Support**: Multiple TTS voices
- [ ] **Analytics Dashboard**: Usage statistics
- [ ] **User Authentication**: Login system
- [ ] **API Documentation**: Swagger/OpenAPI

## 📄 License

This project is part of the Voice Agent Framework. Built with ❤️ for AI innovation.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

**Ready to experience the future of voice AI?** 🎤✨

Start your web UI and begin talking to your AI assistant with natural, human-like voices!
