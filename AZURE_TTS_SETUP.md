# 🎭 Azure TTS Setup for Natural Voices

**Get human-like, natural voices for your Voice Agent!**

## 🚀 Quick Setup (5 minutes)

### 1. Get Azure Free Account
- Go to [Azure Portal](https://portal.azure.com)
- Sign up for free (12 months free, $200 credit)
- No credit card required for free tier

### 2. Create Speech Service
1. **Search for "Speech"** in Azure portal
2. **Click "Create"**
3. **Fill in details:**
   - Resource group: Create new
   - Name: `voice-agent-tts`
   - Region: `East US` (or your preferred region)
   - Pricing tier: `Free F0` (free tier)
4. **Click "Review + Create"**
5. **Click "Create"**

### 3. Get Your API Key
1. **Go to your Speech service**
2. **Click "Keys and Endpoint"**
3. **Copy Key 1** (this is your `AZURE_TTS_KEY`)
4. **Copy Region** (this is your `AZURE_TTS_REGION`)

### 4. Update Your Code
Edit `hackathon_voice_agent.py` and change these lines:

```python
# Azure TTS Configuration
AZURE_TTS_KEY = "your_actual_key_here"  # Paste your key
AZURE_TTS_REGION = "eastus"  # Your region
AZURE_VOICE = "hi-IN-MadhurNeural"  # The voice you want
```

## 🗣️ Available Voices

### **Indian Voices (Hindi):**
- `hi-IN-MadhurNeural` - Male, natural Hindi
- `hi-IN-SwaraNeural` - Female, natural Hindi

### **English Voices:**
- `en-US-JennyNeural` - Female, natural English
- `en-US-GuyNeural` - Male, natural English
- `en-GB-RyanNeural` - Male, British English

### **Other Languages:**
- `ja-JP-NanamiNeural` - Japanese
- `de-DE-KatjaNeural` - German
- `fr-FR-DeniseNeural` - French

## 💡 How to Change Voice

Just change the `AZURE_VOICE` variable:

```python
# For different voices
AZURE_VOICE = "en-US-JennyNeural"  # American female
AZURE_VOICE = "hi-IN-SwaraNeural"  # Hindi female
AZURE_VOICE = "en-GB-RyanNeural"   # British male
```

## 🔧 Test Your Setup

1. **Install requests library:**
   ```bash
   pip install requests
   ```

2. **Run your voice agent:**
   ```bash
   python hackathon_voice_agent.py
   ```

3. **You should see:**
   ```
   ✅ Azure TTS configured for natural voices!
   🗣️ Voice: hi-IN-MadhurNeural
   ```

## 🎯 Benefits for Hackathon

- **🎭 Natural voices** - Not robotic
- **🌍 Multiple languages** - Show diversity
- **🎨 Professional quality** - Impress judges
- **🔧 Easy customization** - Change voices quickly

## 🚨 Troubleshooting

**"Azure TTS not configured":**
- Check your API key is correct
- Verify your region matches
- Ensure Speech service is running

**"TTS error":**
- Check internet connection
- Verify Azure service is active
- Check billing/quotas

## 💰 Cost

- **Free tier**: 500,000 characters/month
- **Pay-as-you-go**: $16 per 1M characters
- **For hackathon**: Free tier is plenty!

**Your voice agent will now have natural, human-like voices!** 🎉
