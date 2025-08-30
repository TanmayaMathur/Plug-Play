# 🎯 Hackathon Voice Agent - Clean Pipeline

**A focused, hackathon-ready voice AI system that does exactly what you need:**

1. **Choose LLM**: Your own pretrained LLM or our inbuilt LLM
2. **Set Context**: If using inbuilt, specify your work domain
3. **Voice Pipeline**: Speech → LLM → Speech (that's it!)

## 🚀 Quick Start (Hackathon Style!)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Voice Agent
```bash
python hackathon_voice_agent.py
```

### 3. Follow the Simple Flow
- Choose: **1** (Your LLM) or **2** (Our LLM)
- If **2**: Enter your work context (e.g., "healthcare", "customer support")
- Start talking! The system will:
  - Record your voice (5 seconds)
  - Process through LLM
  - Give you voice response
  - Ask if you want to continue

## 🎯 What This Does (Exactly)

```
🎤 Your Voice → 🧠 LLM Processing → 🔊 AI Voice Response
```

**No fancy features, no complex menus, no unnecessary complexity.**

## 🔧 For Your Own LLM (Option 1)

When you choose "Use your own pretrained LLM":
- Enter your API endpoint
- Enter your API key (if needed)
- Your LLM handles the thinking part
- Our framework handles STT and TTS

## 🤖 For Our Inbuilt LLM (Option 2)

When you choose "Use our inbuilt LLM":
- Enter your work context/domain
- We use GPT-3.5 Turbo via OpenRouter
- Context helps AI understand your use case
- Perfect for demos and prototypes

## 🎤 How It Works

1. **Record**: 5-second voice recording
2. **Transcribe**: Whisper converts speech to text
3. **Process**: LLM generates response
4. **Synthesize**: pyttsx3 converts text to speech
5. **Output**: AI speaks the response

## 💡 Hackathon Tips

- **Keep it simple**: This is designed for demos
- **Test quickly**: 5-second recordings for fast iteration
- **Show the pipeline**: Demonstrate STT → LLM → TTS flow
- **Customize context**: Change context to match your demo
- **Use your own LLM**: Show integration capabilities

## 🚨 If Something Breaks

**Common issues:**
- **PyAudio error**: `pip install pyaudio --only-binary=all`
- **API key error**: Check your OpenRouter key
- **Import error**: Make sure you're in the right directory

## 🎉 Demo Script

**Perfect for hackathon presentation:**

1. "This is a voice AI framework that anyone can use"
2. "Users choose between their own LLM or our inbuilt one"
3. "Watch this complete pipeline: Speech → AI → Speech"
4. "It's modular - you can plug in any LLM you want"
5. "Ready for production use in any industry"

## 🏆 Why This Wins Hackathons

- ✅ **Working prototype** in minutes
- ✅ **Real voice interaction** (not just text)
- ✅ **Modular architecture** (shows technical skill)
- ✅ **Production ready** (not just a demo)
- ✅ **Industry applicable** (real business value)
- ✅ **Easy to explain** (clear pipeline)

**This is exactly what judges want to see: a working system that solves a real problem!** 🚀
