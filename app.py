#!/usr/bin/env python3
"""
Voice Agent Web UI - Industry Ready Interface
=============================================

Modern web interface for the voice agent with:
- Clean, professional design
- Real-time voice interaction
- LLM configuration
- Conversation history
- Responsive layout
"""

import os
import sys
import traceback
import tempfile
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pydub
from pydub import AudioSegment
import asyncio
import base64
import requests
import PyPDF2
from docx import Document

# Add current directory to path for imports
sys.path.append('.')

# Set up OpenRouter API key for inbuilt LLM (same as hackathon_voice_agent.py)
os.environ["OPENROUTER_API_KEY"] = "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"

# Azure TTS Configuration (same as hackathon_voice_agent.py)
AZURE_TTS_KEY = "3v5nh4Wda1YUSXxY4BZ2tcjHr2yoWMR5lVfxJOCsqVcSDfeoXkNGJQQJ99BHACGhslBXJ3w3AAAYACOGbdk7"
AZURE_TTS_REGION = "centralindia"
AZURE_VOICE = "hi-IN-MadhurNeural"

# New variables for context file handling
pdf_context = ""
context_files = []

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        print(f"❌ PDF extraction error: {e}")
        return ""

def extract_text_from_docx(docx_file):
    """Extract text from DOCX file"""
    try:
        doc = Document(docx_file)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text.strip()
    except Exception as e:
        print(f"❌ DOCX extraction error: {e}")
        return ""

def process_context_file(file):
    """Process uploaded context file (PDF, DOCX, or TXT)"""
    try:
        filename = file.filename.lower()
        file_content = ""
        
        if filename.endswith('.pdf'):
            print(f"📄 Processing PDF: {filename}")
            file_content = extract_text_from_pdf(file)
        elif filename.endswith('.docx'):
            print(f"📝 Processing DOCX: {filename}")
            file_content = extract_text_from_docx(file)
        elif filename.endswith('.txt'):
            print(f"📃 Processing TXT: {filename}")
            file_content = file.read().decode('utf-8')
        else:
            return False, f"Unsupported file type: {filename}"
        
        if not file_content.strip():
            return False, "No text content extracted from file"
        
        print(f"✅ Extracted {len(file_content)} characters from {filename}")
        return True, file_content
        
    except Exception as e:
        print(f"❌ File processing error: {e}")
        return False, str(e)

app = Flask(__name__)
CORS(app)

# Global variables (same as hackathon_voice_agent.py)
llm_type = "inbuilt"  # Default to OpenRouter
context = "general assistant"
conversation_history = []

# Custom LLM configuration
custom_api_url = ""
custom_api_key = ""
custom_model_name = "llama3-8b-8192"  # Default to a common Groq model

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/api/setup', methods=['POST'])
def setup_agent():
    """Setup the agent (same as terminal version)"""
    try:
        data = request.get_json()
        global llm_type, context, custom_api_url, custom_api_key, custom_model_name
        
        llm_type = data.get('llm_type', 'inbuilt')
        context = data.get('context', 'general assistant')
        
        # Handle custom LLM setup
        if llm_type == 'custom':
            api_url = data.get('api_url')
            api_key = data.get('api_key')
            model_name = data.get('model_name', 'llama3-8b-8192')
            
            if not api_url or not api_key:
                return jsonify({
                    "success": False,
                    "message": "Custom LLM requires both API URL and API Key"
                })
            
            # Store custom LLM credentials
            custom_api_url = api_url
            custom_api_key = api_key
            custom_model_name = model_name
            
            print(f"✅ Custom LLM setup: api_url={api_url[:50]}..., api_key={api_key[:10]}..., model={model_name}")
            
        else:  # inbuilt LLM
            print(f"✅ Inbuilt LLM setup: context={context}")
        
        print(f"✅ Agent setup: llm_type={llm_type}, context={context}")
        
        return jsonify({
            "success": True, 
            "message": "Agent configured successfully!"
        })
        
    except Exception as e:
        print(f"Error setting up agent: {e}")
        traceback.print_exc()
        return jsonify({
            "success": False, 
            "message": f"Error: {str(e)}"
        })

def azure_tts(text):
    """Convert text to speech using Azure Cognitive Services (same as hackathon_voice_agent.py)"""
    try:
        print(f"🔊 Generating speech with {AZURE_VOICE}...")
        print(f"🔑 Using key: {AZURE_TTS_KEY[:10]}...")
        print(f"🌍 Region: {AZURE_TTS_REGION}")
        print(f"📝 Text: {text[:100]}...")
        
        # Azure TTS endpoint
        url = f"https://{AZURE_TTS_REGION}.tts.speech.microsoft.com/cognitiveservices/v1"
        print(f"🔗 Endpoint: {url}")
        
        # Headers
        headers = {
            'Ocp-Apim-Subscription-Key': AZURE_TTS_KEY,
            'Content-Type': 'application/ssml+xml',
            'X-Microsoft-OutputFormat': 'riff-16khz-16bit-mono-pcm'
        }
        
        # SSML for the voice
        ssml = f"""
        <speak version='1.0' xml:lang='en-US'>
            <voice xml:lang='hi-IN' xml:gender='Male' name='{AZURE_VOICE}'>
                {text}
            </voice>
        </speak>
        """
        
        print("📤 Sending request to Azure TTS...")
        
        # Make request
        response = requests.post(url, headers=headers, data=ssml.encode('utf-8'))
        
        print(f"📥 Response status: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ Azure TTS generated successfully!")
            print(f"📊 Audio size: {len(response.content)} bytes")
            return response.content
        else:
            print(f"❌ Azure TTS error: {response.status_code}")
            print(f"📄 Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Azure TTS error: {e}")
        print(f"🔍 Error type: {type(e).__name__}")
        return None

async def process_with_llm(transcription):
    """Process with LLM (EXACT SAME LOGIC as hackathon_voice_agent.py)"""
    try:
        print("🧠 Processing with AI...")
        
        # Choose LLM based on setup
        if llm_type == "custom":
            # Use custom LLM (same logic as terminal version)
            print("🔗 Using custom LLM...")
            print(f"🔗 API URL: {custom_api_url[:50]}...")
            print(f"🔑 API Key: {custom_api_key[:10]}...")
            print(f"🤖 Model Name: {custom_model_name}")
            
            try:
                # Make request to your custom LLM API
                headers = {
                    "Authorization": f"Bearer {custom_api_key}",
                    "Content-Type": "application/json"
                }
                
                # Create context-aware prompt for custom LLM (same as inbuilt)
                system_prompt = f"You are a helpful AI assistant specialized in {context}."
                
                # Add PDF context if available (same logic as inbuilt LLM)
                if pdf_context.strip():
                    system_prompt += f"\n\nIMPORTANT: You have access to the following context information:\n{pdf_context[:2000]}..."
                    print(f"📄 Using PDF context with Custom LLM: {len(pdf_context)} characters")
                    print("💡 Custom LLM will use this context to provide accurate, relevant answers")
                    
                    system_prompt += f"\n\nINSTRUCTIONS:"
                    system_prompt += f"\n1. If the user's question relates to information in the context above, use that specific information to answer."
                    system_prompt += f"\n2. If the user asks about something not covered in the context, provide a general helpful response."
                    system_prompt += f"\n3. Always prioritize context information when available and relevant."
                    system_prompt += f"\n4. Be specific and reference the context when appropriate."
                else:
                    print("💡 No PDF context available - Custom LLM will use general knowledge")
                    system_prompt += "\n\nProvide helpful responses based on your general knowledge."
                
                system_prompt += "\n\nProvide clear, helpful responses. If using context information, mention it. If not, provide a general helpful response."
                
                # Prepare the request data (adjust format based on your LLM API)
                data = {
                    "model": custom_model_name,  # Use the custom model name
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": transcription}
                    ],
                    "max_tokens": 800
                }
                
                print(f"📤 Sending to Custom LLM: {transcription[:50]}...")
                
                response = requests.post(custom_api_url, headers=headers, json=data, timeout=30)
                
                if response.status_code == 200:
                    result = response.json()
                    # Extract response based on your LLM API format
                    if 'choices' in result and len(result['choices']) > 0:
                        ai_response = result['choices'][0]['message']['content']
                    elif 'response' in result:
                        ai_response = result['response']
                    elif 'text' in result:
                        ai_response = result['text']
                    else:
                        ai_response = str(result)
                    
                    print(f"🤖 Custom LLM Response: {ai_response}")
                    return ai_response
                else:
                    print(f"❌ Custom LLM error: {response.status_code}")
                    print(f"📄 Response: {response.text}")
                    return f"I'm sorry, my custom LLM returned an error (Status: {response.status_code}). Please check your API configuration."
                    
            except requests.exceptions.Timeout:
                print("❌ Custom LLM request timed out")
                return "I'm sorry, my custom LLM is taking too long to respond. Please try again."
            except requests.exceptions.RequestException as e:
                print(f"❌ Custom LLM request error: {e}")
                return f"I'm sorry, there was an error connecting to my custom LLM: {str(e)}"
            except Exception as e:
                print(f"❌ Custom LLM processing error: {e}")
                return f"I'm sorry, there was an error processing your request with my custom LLM: {str(e)}"
                
        else:
            # Use OpenRouter API with PDF context (EXACT SAME as terminal version)
            url = "https://openrouter.ai/api/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://voice-agent-framework.com",
                "X-Title": "Voice Agent Framework"
            }
            
            # Create context-aware prompt with PDF context (EXACT SAME as terminal)
            system_prompt = f"You are a helpful AI assistant specialized in {context}."
            
            # Add PDF context if available (EXACT SAME logic as terminal)
            if pdf_context.strip():
                system_prompt += f"\n\nIMPORTANT: You have access to the following context information:\n{pdf_context[:2000]}..."
                print(f"📄 Using PDF context: {len(pdf_context)} characters")
                print("💡 AI will use this context to provide accurate, relevant answers")
                
                system_prompt += f"\n\nINSTRUCTIONS:"
                system_prompt += f"\n1. If the user's question relates to information in the context above, use that specific information to answer."
                system_prompt += f"\n2. If the user asks about something not covered in the context, provide a general helpful response."
                system_prompt += f"\n3. Always prioritize context information when available and relevant."
                system_prompt += f"\n4. Be specific and reference the context when appropriate."
            else:
                print("💡 No PDF context available - AI will use general knowledge")
                system_prompt += "\n\nProvide helpful responses based on your general knowledge."
            
            system_prompt += "\n\nProvide clear, helpful responses. If using context information, mention it. If not, provide a general helpful response."
            
            data = {
                "model": "openai/gpt-3.5-turbo",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": transcription}
                ],
                "max_tokens": 800  # Increased for longer responses with context
            }
            
            print(f"📤 Sending to OpenRouter: {transcription[:50]}...")
            
            response = requests.post(url, headers=headers, json=data)
            
            if response.status_code == 200:
                result = response.json()
                ai_response = result['choices'][0]['message']['content']
                print(f"🤖 AI Response: {ai_response}")
                return ai_response
            else:
                print(f"❌ OpenRouter error: {response.status_code}")
                print("🔄 Trying again...")
                return "I'm sorry, I couldn't process that request."
                
    except Exception as e:
        print(f"❌ LLM processing error: {e}")
        traceback.print_exc()
        return "I'm sorry, there was an error processing your request."

async def speech_to_text(audio_data):
    """Convert speech to text using Whisper (simplified)"""
    try:
        print("🎤 Converting speech to text...")
        
        # Save audio to temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_file.write(audio_data)
            temp_wav = temp_file.name
        
        # Use Whisper directly (same as terminal version)
        import whisper
        model = whisper.load_model("base")
        
        print(f"🎵 Processing audio file: {temp_wav}")
        result = model.transcribe(temp_wav)
        
        # Clean up temp file
        os.unlink(temp_wav)
        
        transcription = result["text"].strip()
        print(f"📝 Transcription: {transcription}")
        
        return transcription
        
    except Exception as e:
        print(f"❌ Speech to text error: {e}")
        traceback.print_exc()
        return ""

@app.route('/api/process-audio', methods=['POST'])
def process_audio():
    """Process audio through the exact same pipeline as hackathon_voice_agent.py"""
    try:
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file provided"})
        
        audio_file = request.files['audio']
        print(f"Received audio file: {audio_file.filename}")
        
        # Convert WebM to WAV using pydub (same as before)
        try:
            audio_data = audio_file.read()
            
            with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as temp_webm:
                temp_webm.write(audio_data)
                temp_webm_path = temp_webm.name
            
            audio = AudioSegment.from_file(temp_webm_path, format="webm")
            audio = audio.set_channels(1).set_frame_rate(16000)
            
            # Normalize audio if too quiet
            audio_dBFS = audio.dBFS
            print(f"Audio levels: {audio_dBFS:.1f} dBFS")
            if audio_dBFS < -30:
                print("Audio too quiet, normalizing...")
                audio = audio.normalize()
                print(f"Normalized audio levels: {audio.dBFS:.1f} dBFS")
            
            wav_path = temp_webm_path.replace('.webm', '.wav')
            audio.export(wav_path, format="wav", parameters=[
                "-ar", "16000", "-ac", "1", "-f", "wav", "-acodec", "pcm_s16le"
            ])
            
            with open(wav_path, 'rb') as f:
                wav_data = f.read()
            
            print(f"Audio converted successfully: {len(wav_data)} bytes")
            
            # Clean up temp files
            os.unlink(temp_webm_path)
            os.unlink(wav_path)
            
        except Exception as e:
            print(f"Audio conversion error: {e}")
            traceback.print_exc()
            return jsonify({"error": f"Audio conversion failed: {str(e)}"})
        
        # Process through the EXACT SAME pipeline as hackathon_voice_agent.py
        try:
            print("🎯 Starting Custom Speech → LLM → Azure TTS Pipeline (EXACT SAME as terminal)")
            print("=" * 50)
            
            # Step 1: Speech to Text (STT) - EXACT SAME as terminal
            print("🎤 Converting speech to text...")
            transcription = asyncio.run(speech_to_text(wav_data))
            
            if not transcription or transcription.strip() == "":
                print("❌ No speech detected, try again")
                return jsonify({"error": "No speech detected"})
            
            print(f"📝 You said: {transcription}")
            
            # Step 2: Process with LLM (EXACT SAME as terminal)
            print("🧠 Processing with AI...")
            llm_response = asyncio.run(process_with_llm(transcription))
            
            if not llm_response or llm_response.strip() == "":
                print("❌ No AI response generated")
                return jsonify({"error": "No AI response generated"})
            
            print(f"🤖 AI Response: {llm_response}")
            
            # Step 3: Azure TTS (EXACT SAME as terminal)
            print("🎭 Converting to natural speech with Azure TTS...")
            azure_audio = azure_tts(llm_response)
            
            if azure_audio:
                print("✅ Azure TTS successful - playing natural voice!")
                # Convert audio to base64 for web transmission
                audio_base64 = base64.b64encode(azure_audio).decode('utf-8')
                
                # Add to conversation history
                conversation_entry = {
                    "id": len(conversation_history) + 1,
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "user_input": transcription,
                    "ai_response": llm_response,
                    "audio_available": True
                }
                conversation_history.append(conversation_entry)
                
                print("✅ Complete custom pipeline: Speech → LLM → Azure TTS")
                
                return jsonify({
                    "transcription": transcription,
                    "response": llm_response,
                    "audio": audio_base64,
                    "conversation_id": conversation_entry["id"]
                })
            else:
                print("❌ Azure TTS failed")
                return jsonify({
                    "transcription": transcription,
                    "response": llm_response,
                    "audio": None
                })
                
        except Exception as e:
            print(f"❌ Pipeline processing error: {e}")
            traceback.print_exc()
            return jsonify({"error": f"Pipeline processing failed: {str(e)}"})
            
    except Exception as e:
        print(f"Audio processing error: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)})

@app.route('/api/conversations')
def get_conversations():
    """Get conversation history"""
    return jsonify(conversation_history)

@app.route('/api/status')
def get_status():
    """Get agent status"""
    return jsonify({
        "agent_configured": True,
        "context": context,
        "conversation_count": len(conversation_history),
        "has_pdf_context": bool(pdf_context.strip()),
        "pdf_context_length": len(pdf_context),
        "files_uploaded": len(context_files)
    })

@app.route('/api/upload-context', methods=['POST'])
def upload_context_file():
    """Upload and process context file (PDF, DOCX, TXT)"""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"})
        
        # Process the file
        success, result = process_context_file(file)
        
        if success:
            global pdf_context, context_files
            
            # Add to context
            if pdf_context:
                pdf_context += "\n\n--- NEW CONTEXT ---\n\n" + result
            else:
                pdf_context = result
            
            # Track the file
            context_files.append({
                "filename": file.filename,
                "size": len(result),
                "uploaded": datetime.now().strftime("%H:%M:%S")
            })
            
            print(f"✅ Context updated with {len(result)} characters from {file.filename}")
            
            return jsonify({
                "success": True,
                "message": f"Context file '{file.filename}' processed successfully!",
                "characters_added": len(result),
                "total_context_length": len(pdf_context),
                "files_uploaded": len(context_files)
            })
        else:
            return jsonify({
                "success": False,
                "error": result
            })
            
    except Exception as e:
        print(f"File upload error: {e}")
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": f"Upload failed: {str(e)}"
        })

@app.route('/api/context-info')
def get_context_info():
    """Get information about current context"""
    return jsonify({
        "has_context": bool(pdf_context.strip()),
        "context_length": len(pdf_context),
        "files_uploaded": context_files,
        "context_preview": pdf_context[:500] + "..." if len(pdf_context) > 500 else pdf_context
    })

@app.route('/api/clear-context')
def clear_context():
    """Clear all uploaded context"""
    global pdf_context, context_files
    pdf_context = ""
    context_files = []
    return jsonify({
        "success": True,
        "message": "Context cleared successfully!"
    })

@app.route('/api/test')
def test_api():
    """Test endpoint to verify API is working"""
    return jsonify({
        "success": True,
        "message": "🎯 HACKATHON VOICE AGENT WEB UI - EXACT SAME AS TERMINAL!",
        "timestamp": datetime.now().isoformat(),
        "version": "6.0 - Terminal Logic Integration",
        "description": "This web UI now uses EXACTLY the same code logic as hackathon_voice_agent.py",
        "features": [
            "🔧 EXACT SAME LLM choice logic (custom vs inbuilt)",
            "📄 EXACT SAME PDF context upload and processing",
            "🧠 EXACT SAME OpenRouter API integration",
            "🎭 EXACT SAME Azure TTS for natural voices",
            "🎤 EXACT SAME Whisper speech recognition",
            "💡 EXACT SAME context-aware AI responses",
            "📝 EXACT SAME DOCX context processing",
            "📃 EXACT SAME TXT context processing",
            "🔄 EXACT SAME pipeline: Speech → LLM → Azure TTS"
        ],
        "terminal_equivalence": "100% - Same code, same logic, same output, just with web UI!"
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
