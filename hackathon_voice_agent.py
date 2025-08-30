#!/usr/bin/env python3
"""
Hackathon Voice Agent - Clean Pipeline
======================================

Simple, focused voice agent for hackathon demo:
1. Choose LLM: Own pretrained or Inbuilt
2. If inbuilt: Set context/domain
3. Speech → LLM → Speech pipeline
4. Natural human-like TTS voices
"""

import os
import sys
import asyncio
import time
import pyaudio
import wave
import tempfile
import requests
import PyPDF2
from docx import Document
import base64
import traceback
sys.path.append('.')

# Set up OpenRouter API key for inbuilt LLM
os.environ["OPENROUTER_API_KEY"] = "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"

# Azure TTS Configuration (you can change these)
AZURE_TTS_KEY = "3v5nh4Wda1YUSXxY4BZ2tcjHr2yoWMR5lVfxJOCsqVcSDfeoXkNGJQQJ99BHACGhslBXJ3w3AAAYACOGbdk7"
AZURE_TTS_REGION = "centralindia"
AZURE_VOICE = "hi-IN-MadhurNeural"  # The voice you want

# Global context storage for terminal version
pdf_context = ""
context_files = []

def extract_text_from_pdf(pdf_file_path):
    """Extract text from PDF file"""
    try:
        with open(pdf_file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        print(f"❌ PDF extraction error: {e}")
        return ""

def extract_text_from_docx(docx_file_path):
    """Extract text from DOCX file"""
    try:
        doc = Document(docx_file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text.strip()
    except Exception as e:
        print(f"❌ DOCX extraction error: {e}")
        return ""

def extract_text_from_txt(txt_file_path):
    """Extract text from TXT file"""
    try:
        with open(txt_file_path, 'r', encoding='utf-8') as file:
            return file.read().strip()
    except Exception as e:
        print(f"❌ TXT extraction error: {e}")
        return ""

def process_context_file(file_path):
    """Process context file (PDF, DOCX, or TXT)"""
    try:
        filename = file_path.lower()
        file_content = ""
        
        if filename.endswith('.pdf'):
            print(f"📄 Processing PDF: {file_path}")
            file_content = extract_text_from_pdf(file_path)
        elif filename.endswith('.docx'):
            print(f"📝 Processing DOCX: {file_path}")
            file_content = extract_text_from_docx(file_path)
        elif filename.endswith('.txt'):
            print(f"📃 Processing TXT: {file_path}")
            file_content = extract_text_from_txt(file_path)
        else:
            return False, f"Unsupported file type: {filename}"
        
        if not file_content.strip():
            return False, "No text content extracted from file"
        
        print(f"✅ Extracted {len(file_content)} characters from {file_path}")
        return True, file_content
        
    except Exception as e:
        print(f"❌ File processing error: {e}")
        return False, str(e)

def get_file_path_interactive():
    """Interactive file path input with suggestions"""
    print("\n📁 FILE PATH INPUT")
    print("=" * 30)
    print("Choose input method:")
    print("1. 📝 Type full path manually")
    print("2. 🔍 Browse current directory")
    print("3. 📂 Enter relative path from current directory")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == "1":
        # Manual path input
        file_path = input("Enter full file path: ").strip()
        return file_path
    
    elif choice == "2":
        # Browse current directory
        current_dir = os.getcwd()
        print(f"\n📂 Current directory: {current_dir}")
        print("📄 Available files:")
        
        # List PDF, DOCX, and TXT files
        supported_files = []
        for file in os.listdir(current_dir):
            if file.lower().endswith(('.pdf', '.docx', '.txt')):
                supported_files.append(file)
                print(f"   📄 {file}")
        
        if not supported_files:
            print("❌ No supported files found in current directory")
            return None
        
        file_name = input("Enter filename from list above: ").strip()
        if file_name in supported_files:
            return os.path.join(current_dir, file_name)
        else:
            print("❌ Invalid filename")
            return None
    
    elif choice == "3":
        # Relative path
        current_dir = os.getcwd()
        print(f"\n📂 Current directory: {current_dir}")
        relative_path = input("Enter relative path (e.g., docs/file.pdf): ").strip()
        full_path = os.path.join(current_dir, relative_path)
        return full_path
    
    else:
        print("❌ Invalid choice")
        return None

class HackathonVoiceAgent:
    def __init__(self):
        self.llm_type = None
        self.context = None
        self.agent = None
        
    def get_user_choice(self):
        """Get user's LLM choice and context"""
        print("🎯 HACKATHON VOICE AGENT")
        print("=" * 50)
        print("Choose your LLM option:")
        print("1. 🔗 Use your own pretrained LLM")
        print("2. 🤖 Use our inbuilt LLM (OpenRouter)")
        print()
        
        while True:
            choice = input("Enter choice (1 or 2): ").strip()
            if choice == "1":
                self.llm_type = "custom"
                print("✅ You chose: Your own pretrained LLM")
                print("💡 No context upload needed for custom LLM")
                break
            elif choice == "2":
                self.llm_type = "inbuilt"
                print("✅ You chose: Our inbuilt LLM (OpenRouter)")
                print("📄 Context upload is recommended for better responses")
                break
            else:
                print("❌ Please enter 1 or 2")
        
        # If inbuilt LLM is chosen, ask for context upload
        if self.llm_type == "inbuilt":
            print("\n📝 Set your work context/domain:")
            print("Examples: 'customer support', 'healthcare', 'education', 'finance'")
            self.context = input("Enter context: ").strip()
            if not self.context:
                self.context = "general assistant"
            print(f"✅ Context set: {self.context}")
            
            # Ask if user wants to upload context files
            print("\n📄 Would you like to upload context files (PDF/DOCX/TXT)?")
            print("This will help the AI provide more accurate and relevant responses.")
            upload_choice = input("Upload context files? (y/N): ").strip().lower()
            
            if upload_choice == 'y':
                self.upload_context_file()
        
        # Show context management options for inbuilt LLM
        if self.llm_type == "inbuilt":
            while True:
                print("\n📋 Context Management:")
                print("1. 📄 Upload More Context Files")
                print("2. 📋 View Current Context")
                print("3. 🗑️ Clear Context")
                print("4. ▶️ Continue to Voice Agent")
                
                context_choice = input("Enter choice (1-4): ").strip()
                if context_choice == "1":
                    self.upload_context_file()
                elif context_choice == "2":
                    self.view_context()
                elif context_choice == "3":
                    self.clear_context()
                elif context_choice == "4":
                    print("✅ Proceeding to Voice Agent setup...")
                    break
                else:
                    print("❌ Please enter 1-4")
    
    def upload_context_file(self):
        """Upload context file from terminal"""
        print("\n📄 CONTEXT FILE UPLOAD")
        print("=" * 30)
        print("Supported formats: PDF, DOCX, TXT")
        
        # Get file path interactively
        file_path = get_file_path_interactive()
        
        if not file_path:
            print("❌ No file path provided")
            return
        
        # Fix Windows path issues
        file_path = file_path.replace('\\', '/')  # Convert backslashes to forward slashes
        file_path = os.path.normpath(file_path)  # Normalize the path
        
        print(f"🔍 Checking file: {file_path}")
        
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            print("💡 Please check:")
            print("   - File path is correct")
            print("   - File exists in the specified location")
            print("   - Use forward slashes (/) in the path")
            print("   - Example: C:/Python/tq.pdf")
            return
        
        # Process the file
        success, result = process_context_file(file_path)
        
        if success:
            global pdf_context, context_files
            
            # Add to context
            if pdf_context:
                pdf_context += "\n\n--- NEW CONTEXT ---\n\n" + result
            else:
                pdf_context = result
            
            # Track the file
            context_files.append({
                "filename": os.path.basename(file_path),
                "size": len(result),
                "uploaded": time.strftime("%H:%M:%S")
            })
            
            print(f"✅ Context updated with {len(result)} characters from {os.path.basename(file_path)}")
            print(f"📊 Total context length: {len(pdf_context)} characters")
        else:
            print(f"❌ Failed to process file: {result}")
    
    def view_context(self):
        """View current context information"""
        print("\n📋 CURRENT CONTEXT")
        print("=" * 30)
        
        if pdf_context.strip():
            print(f"📄 Context loaded: {len(pdf_context)} characters")
            print(f"📁 Files uploaded: {len(context_files)}")
            
            for i, file_info in enumerate(context_files, 1):
                print(f"  {i}. {file_info['filename']} ({file_info['size']} chars) - {file_info['uploaded']}")
            
            print(f"\n📖 Context Preview (first 300 chars):")
            print("-" * 40)
            print(pdf_context[:300] + "..." if len(pdf_context) > 300 else pdf_context)
            print("-" * 40)
        else:
            print("❌ No context files uploaded yet")
    
    def clear_context(self):
        """Clear all uploaded context"""
        global pdf_context, context_files
        
        if not pdf_context.strip():
            print("❌ No context to clear")
            return
        
        confirm = input("Are you sure you want to clear all context? (y/N): ").strip().lower()
        if confirm == 'y':
            pdf_context = ""
            context_files = []
            print("✅ All context cleared successfully!")
        else:
            print("Context clearing cancelled")
    
    def setup_agent(self, llm_type=None, api_url=None, api_key=None, context=None):
        """Set up the voice agent based on user choice"""
        print("\n🔧 Setting up voice agent...")
        
        # Use provided parameters or defaults
        if llm_type is None:
            llm_type = self.llm_type
        if context is None:
            context = self.context
            
        print(f"Setting up agent with: llm_type={llm_type}, context={context}")
        
        try:
            from voice_agent import VoiceAgent
            self.agent = VoiceAgent()
            
            if llm_type == "custom":
                # Custom LLM setup
                print("🔗 Setting up custom LLM...")
                if not api_url:
                    print("❌ Custom LLM requires API URL")
                    return False
                
                self.agent.configure_llm(
                    provider="custom",
                    api_url=api_url,
                    api_key=api_key,
                    model="custom",
                    agent_type="custom"
                )
                print("✅ Custom LLM configured!")
                
            else:  # inbuilt - using OpenRouter
                # OpenRouter LLM setup
                print("🤖 Setting up OpenRouter LLM...")
                self.agent.configure_llm(
                    provider="custom",  # Use custom provider for OpenRouter
                    api_url="https://openrouter.ai/api/v1/chat/completions",
                    api_key=os.environ["OPENROUTER_API_KEY"],
                    model="openai/gpt-3.5-turbo",
                    agent_type="custom",
                    custom_payload_format="openai",
                    custom_headers={
                        "HTTP-Referer": "https://voice-agent-framework.com",
                        "X-Title": "Voice Agent Framework"
                    }
                )
                print("✅ OpenRouter LLM configured!")
                
        except Exception as e:
            print(f"❌ Error setting up agent: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        return True
    
    def record_audio(self, duration=5):
        """Record audio from microphone"""
        print(f"🎤 Recording {duration} seconds... (speak now!)")
        
        # Audio settings
        sample_rate = 16000
        chunk_size = 1024
        total_chunks = int(sample_rate * duration / chunk_size)
        
        # Initialize PyAudio
        audio = pyaudio.PyAudio()
        stream = audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=sample_rate,
            input=True,
            frames_per_buffer=chunk_size
        )
        
        frames = []
        
        # Record audio
        for i in range(total_chunks):
            data = stream.read(chunk_size)
            frames.append(data)
        
        # Stop recording
        stream.stop_stream()
        stream.close()
        audio.terminate()
        
        # Combine audio frames
        audio_data = b''.join(frames)
        
        # Save to WAV file for Whisper
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            wav_file = temp_file.name
        
        with wave.open(wav_file, 'wb') as wav:
            wav.setnchannels(1)
            wav.setframerate(sample_rate)
            wav.setsampwidth(2)
            wav.writeframes(audio_data)
        
        # Read WAV file
        with open(wav_file, 'rb') as f:
            wav_data = f.read()
        
        # Clean up temp file
        os.unlink(wav_file)
        
        return wav_data
    
    def azure_tts(self, text):
        """Convert text to speech using Azure Cognitive Services"""
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
    
    def play_audio(self, audio_data):
        """Play audio through speakers"""
        try:
            print("🔊 Playing AI response...")
            
            # Save audio to temporary WAV file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                wav_file = temp_file.name
            
            # Write audio data to WAV file
            with open(wav_file, 'wb') as f:
                f.write(audio_data)
            
            # Play the audio
            audio = pyaudio.PyAudio()
            stream = audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=16000,  # Azure TTS rate
                output=True,
                frames_per_buffer=1024
            )
            
            # Read and play the WAV file
            with wave.open(wav_file, 'rb') as wav:
                data = wav.readframes(1024)
                while data:
                    stream.write(data)
                    data = wav.readframes(1024)
            
            # Clean up
            stream.stop_stream()
            stream.close()
            audio.terminate()
            os.unlink(wav_file)
            
            print("✅ Audio played successfully!")
            
        except Exception as e:
            print(f"❌ Error playing audio: {e}")
    
    async def process_speech_pipeline(self):
        """Process speech through the complete pipeline"""
        print("\n🎯 Starting Speech → LLM → Speech Pipeline")
        print("=" * 50)
        
        while True:
            try:
                # Record speech
                audio_data = self.record_audio(5)
                
                # Process through pipeline
                print("🔄 Processing your voice...")
                start_time = time.time()
                
                # Get the result from the agent
                result = await self.agent.process_audio(audio_data)
                end_time = time.time()
                
                # Extract transcription and response
                transcription = ""
                response = ""
                
                if isinstance(result, dict):
                    transcription = result.get('transcription', 'No speech detected')
                    response = result.get('response', 'No response')
                else:
                    # If result is bytes, we need to get the text from the agent directly
                    print("🔍 Getting response text from agent...")
                    
                    # Try to get the last conversation from the agent's conversation history
                    if hasattr(self.agent, 'llm_agent'):
                        # Get the conversation history
                        if hasattr(self.agent.llm_agent, 'conversation_history') and self.agent.llm_agent.conversation_history:
                            # Get the last conversation
                            last_conversation = self.agent.llm_agent.conversation_history[-1]
                            print(f"🔍 Last conversation: {last_conversation}")
                            
                            # Try different possible keys
                            transcription = (last_conversation.get('user_input') or 
                                          last_conversation.get('input') or 
                                          last_conversation.get('message') or 
                                          'No speech detected')
                            
                            response = (last_conversation.get('response') or 
                                      last_conversation.get('output') or 
                                      last_conversation.get('reply') or 
                                      'No response')
                        else:
                            print("❌ No conversation history found")
                            print("🔍 Available attributes:", dir(self.agent.llm_agent))
                            continue
                    else:
                        print("❌ Cannot access llm_agent")
                        continue
                
                # Display what we got
                print(f"📝 You said: {transcription}")
                print(f"🤖 AI Response: {response}")
                
                # Use Azure TTS for natural voice
                if response and response != "No response" and len(response) > 10:
                    print("🎭 Using Azure TTS for natural voice...")
                    azure_audio = self.azure_tts(response)
                    if azure_audio:
                        print("✅ Azure TTS successful - playing natural voice!")
                        self.play_audio(azure_audio)
                    else:
                        print("❌ Azure TTS failed - please check your configuration")
                        print("💡 Make sure your Azure key and region are correct")
                        continue
                else:
                    print("❌ No valid response to convert to speech")
                    print(f"🔍 Response length: {len(response) if response else 0}")
                    continue
                
                print(f"⏱️  Total time: {end_time - start_time:.2f} seconds")
                print("✅ Complete pipeline: Speech → LLM → Natural TTS")
                
                # Ask if continue
                print("\n" + "-" * 50)
                continue_choice = input("Continue conversation? (y/n): ").strip().lower()
                if continue_choice != 'y':
                    break
                    
            except KeyboardInterrupt:
                print("\n🛑 Stopping...")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                print("🔄 Trying again...")
                continue
    
    async def custom_speech_pipeline(self):
        """Custom pipeline that bypasses framework TTS and uses only Azure TTS"""
        print("\n🎯 Starting Custom Speech → LLM → Azure TTS Pipeline")
        print("=" * 50)
        
        while True:
            try:
                # Record speech
                audio_data = self.record_audio(5)
                
                # Process through pipeline
                print("🔄 Processing your voice...")
                start_time = time.time()
                
                # Step 1: Speech to Text (STT)
                print("🎤 Converting speech to text...")
                transcription = await self.agent._speech_to_text(audio_data)
                print(f"📝 You said: {transcription}")
                
                if not transcription or transcription.strip() == "":
                    print("❌ No speech detected, try again")
                    continue
                
                # Step 2: Process with LLM (enhanced with PDF context)
                print("🧠 Processing with AI...")
                
                # Use OpenRouter API with PDF context instead of framework LLM
                url = "https://openrouter.ai/api/v1/chat/completions"
                headers = {
                    "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://voice-agent-framework.com",
                    "X-Title": "Voice Agent Framework"
                }
                
                # Create context-aware prompt with PDF context
                system_prompt = f"You are a helpful AI assistant specialized in {self.context}."
                
                # Add PDF context if available
                global pdf_context
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
                    llm_response = result['choices'][0]['message']['content']
                    print(f"🤖 AI Response: {llm_response}")
                else:
                    print(f"❌ OpenRouter error: {response.status_code}")
                    print("🔄 Trying again...")
                    continue
                
                # Step 3: Azure TTS
                print("🎭 Converting to natural speech with Azure TTS...")
                azure_audio = self.azure_tts(llm_response)
                
                if azure_audio:
                    print("✅ Azure TTS successful - playing natural voice!")
                    self.play_audio(azure_audio)
                else:
                    print("❌ Azure TTS failed")
                    continue
                
                end_time = time.time()
                print(f"⏱️  Total time: {end_time - start_time:.2f} seconds")
                print("✅ Complete custom pipeline: Speech → LLM → Azure TTS")
                
                # Ask if continue
                print("\n" + "-" * 50)
                continue_choice = input("Continue conversation? (y/n): ").strip().lower()
                if continue_choice != 'y':
                    break
                    
            except KeyboardInterrupt:
                print("\n🛑 Stopping...")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                print("🔄 Trying again...")
                continue
    
    async def custom_speech_pipeline_single(self, audio_data):
        """Process a single audio input through the pipeline for web UI"""
        try:
            print("🎤 Processing single audio input...")
            
            # Step 1: Speech to Text (STT)
            print("🎤 Converting speech to text...")
            transcription = await self.agent._speech_to_text(audio_data)
            print(f"📝 You said: {transcription}")
            
            if not transcription or transcription.strip() == "":
                print("❌ No speech detected")
                return {"error": "No speech detected"}
            
            # Step 2: Process with LLM (enhanced with PDF context)
            print("🧠 Processing with AI...")
            
            # Use OpenRouter API with PDF context
            url = "https://openrouter.ai/api/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://voice-agent-framework.com",
                "X-Title": "Voice Agent Framework"
            }
            
            # Create context-aware prompt with PDF context
            system_prompt = f"You are a helpful AI assistant specialized in {self.context}."
            
            # Add PDF context if available
            global pdf_context
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
                llm_response = result['choices'][0]['message']['content']
                print(f"🤖 AI Response: {llm_response}")
            else:
                print(f"❌ OpenRouter error: {response.status_code}")
                return {"error": "Failed to get AI response"}
            
            # Step 3: Azure TTS
            print("🎭 Converting to natural speech with Azure TTS...")
            azure_audio = self.azure_tts(llm_response)
            
            if azure_audio:
                print("✅ Azure TTS successful!")
                # Convert audio to base64 for web transmission
                audio_base64 = base64.b64encode(azure_audio).decode('utf-8')
                return {
                    "transcription": transcription,
                    "response": llm_response,
                    "audio": audio_base64
                }
            else:
                print("❌ Azure TTS failed")
                return {
                    "transcription": transcription,
                    "response": llm_response,
                    "audio": None
                }
                
        except Exception as e:
            print(f"❌ Pipeline processing error: {e}")
            traceback.print_exc()
            return {"error": str(e)}
    
    def run(self):
        """Main run method"""
        print("🚀 Starting Hackathon Voice Agent...")
        print("🎭 Using Azure TTS for natural human-like voices!")
        print(f"🗣️  Voice: {AZURE_VOICE}")
        print("📄 NEW: PDF Context Upload Support!")
        print("   - Upload PDF, DOCX, or TXT files")
        print("   - AI uses context for better responses")
        print("   - Manage context from terminal menu")
        
        # Check Azure TTS setup
        if AZURE_TTS_KEY == "your_azure_key_here":
            print("⚠️  Azure TTS not configured - using default TTS")
            print("💡 To use natural voices, get Azure key from:")
            print("   https://portal.azure.com/#create/Microsoft.CognitiveServices")
        else:
            print("✅ Azure TTS configured for natural voices!")
        
        print()
        
        # Get user choice
        self.get_user_choice()
        
        # Setup agent
        if not self.setup_agent():
            print("❌ Failed to setup agent. Exiting.")
            return
        
        print("\n✅ Voice Agent is ready!")
        print("🎤 You can now start speaking!")
        print("💡 The system will:")
        print("   1. Record your voice (5 seconds)")
        print("   2. Convert speech to text")
        print("   3. Process through AI (OpenRouter)")
        print("   4. Use PDF context if available")
        print("   5. Speak the response with natural voice 🔊")
        print()
        
        # Start pipeline
        try:
            print("\n🎭 Starting Azure TTS Pipeline!")
            print("=" * 50)
            asyncio.run(self.custom_speech_pipeline())
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
        except Exception as e:
            print(f"❌ Fatal Error: {e}")

def main():
    print("🎯 HACKATHON VOICE AGENT")
    print("=" * 50)
    print("Clean, focused voice AI pipeline:")
    print("• Choose your LLM (own or inbuilt)")
    print("• If inbuilt: Set context + upload PDF/DOCX/TXT files")
    print("• If custom: Skip context, use your own LLM")
    print("• Speech → LLM → Speech pipeline")
    print("• Natural human-like TTS voices")
    print("• 🧠 Context-aware AI responses (for inbuilt LLM)")
    print()
    
    agent = HackathonVoiceAgent()
    agent.run()

if __name__ == "__main__":
    main()
