"""
Web interface for the Voice Agent Framework.
"""

import asyncio
import json
import base64
from typing import Dict, Any, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn

from .core import VoiceAgent
from .agents import BasicAgent, HealthcareAgent, CustomerSupportAgent
from .config import VoiceAgentConfig
from loguru import logger


# Pydantic models for API requests/responses
class AgentConfig(BaseModel):
    stt_engine: str = "whisper"
    tts_engine: str = "elevenlabs"
    agent_type: str = "basic"
    config: Optional[Dict[str, Any]] = None


class AudioRequest(BaseModel):
    audio_data: str  # Base64 encoded audio
    session_id: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    components: Dict[str, str]
    timestamp: str


# FastAPI app
app = FastAPI(
    title="Voice Agent Framework",
    description="A modular voice agent framework with STT → LLM → TTS pipeline",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global agent instances
agents: Dict[str, VoiceAgent] = {}


def create_agent(agent_type: str, config: Optional[Dict[str, Any]] = None) -> VoiceAgent:
    """Create a voice agent based on type."""
    agent_map = {
        "basic": BasicAgent,
        "healthcare": HealthcareAgent,
        "customer_support": CustomerSupportAgent
    }
    
    if agent_type not in agent_map:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    agent_class = agent_map[agent_type]
    agent_instance = agent_class(config or {})
    
    return VoiceAgent(
        stt_engine="whisper",
        llm_agent=agent_instance,
        tts_engine="elevenlabs",
        config=config
    )


@app.get("/", response_class=HTMLResponse)
async def get_index():
    """Serve the main HTML interface."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Voice Agent Framework</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .container { max-width: 800px; margin: 0 auto; }
            .status { padding: 10px; margin: 10px 0; border-radius: 5px; }
            .status.connected { background-color: #d4edda; color: #155724; }
            .status.disconnected { background-color: #f8d7da; color: #721c24; }
            .controls { margin: 20px 0; }
            button { padding: 10px 20px; margin: 5px; border: none; border-radius: 5px; cursor: pointer; }
            button.primary { background-color: #007bff; color: white; }
            button.danger { background-color: #dc3545; color: white; }
            button.success { background-color: #28a745; color: white; }
            .log { background-color: #f8f9fa; padding: 10px; border-radius: 5px; height: 300px; overflow-y: auto; }
            .agent-selector { margin: 20px 0; }
            select { padding: 8px; margin: 5px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎤 Voice Agent Framework</h1>
            
            <div class="agent-selector">
                <label for="agent-type">Agent Type:</label>
                <select id="agent-type">
                    <option value="basic">Basic Q&A</option>
                    <option value="healthcare">Healthcare</option>
                    <option value="customer_support">Customer Support</option>
                </select>
                <button onclick="initializeAgent()" class="primary">Initialize Agent</button>
            </div>
            
            <div id="status" class="status disconnected">
                Disconnected
            </div>
            
            <div class="controls">
                <button onclick="connect()" class="primary">Connect</button>
                <button onclick="disconnect()" class="danger">Disconnect</button>
                <button onclick="startConversation()" class="success">Start Conversation</button>
                <button onclick="stopConversation()" class="danger">Stop Conversation</button>
                <button onclick="clearLog()" class="primary">Clear Log</button>
            </div>
            
            <div>
                <h3>Conversation Log</h3>
                <div id="log" class="log"></div>
            </div>
            
            <div>
                <h3>Performance Stats</h3>
                <div id="stats"></div>
            </div>
        </div>
        
        <script>
            let ws = null;
            let isRecording = false;
            let mediaRecorder = null;
            let audioChunks = [];
            
            function log(message) {
                const logDiv = document.getElementById('log');
                const timestamp = new Date().toLocaleTimeString();
                logDiv.innerHTML += `[${timestamp}] ${message}<br>`;
                logDiv.scrollTop = logDiv.scrollHeight;
            }
            
            function updateStatus(connected) {
                const statusDiv = document.getElementById('status');
                if (connected) {
                    statusDiv.className = 'status connected';
                    statusDiv.textContent = 'Connected';
                } else {
                    statusDiv.className = 'status disconnected';
                    statusDiv.textContent = 'Disconnected';
                }
            }
            
            async function initializeAgent() {
                const agentType = document.getElementById('agent-type').value;
                try {
                    const response = await fetch('/api/agent/initialize', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ agent_type: agentType })
                    });
                    
                    if (response.ok) {
                        const result = await response.json();
                        log(`Agent initialized: ${agentType}`);
                    } else {
                        log(`Failed to initialize agent: ${response.statusText}`);
                    }
                } catch (error) {
                    log(`Error initializing agent: ${error}`);
                }
            }
            
            function connect() {
                if (ws) {
                    ws.close();
                }
                
                ws = new WebSocket('ws://localhost:8000/ws');
                
                ws.onopen = function() {
                    updateStatus(true);
                    log('WebSocket connected');
                };
                
                ws.onmessage = function(event) {
                    const data = JSON.parse(event.data);
                    log(`Response: ${data.response}`);
                    playAudio(data.audio_data);
                };
                
                ws.onclose = function() {
                    updateStatus(false);
                    log('WebSocket disconnected');
                };
                
                ws.onerror = function(error) {
                    log(`WebSocket error: ${error}`);
                };
            }
            
            function disconnect() {
                if (ws) {
                    ws.close();
                    ws = null;
                }
            }
            
            async function startConversation() {
                if (!ws || ws.readyState !== WebSocket.OPEN) {
                    log('WebSocket not connected');
                    return;
                }
                
                try {
                    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                    mediaRecorder = new MediaRecorder(stream);
                    
                    mediaRecorder.ondataavailable = function(event) {
                        audioChunks.push(event.data);
                    };
                    
                    mediaRecorder.onstop = function() {
                        const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                        const reader = new FileReader();
                        
                        reader.onload = function() {
                            const audioData = reader.result.split(',')[1]; // Remove data URL prefix
                            ws.send(JSON.stringify({
                                type: 'audio',
                                data: audioData
                            }));
                        };
                        
                        reader.readAsDataURL(audioBlob);
                        audioChunks = [];
                    };
                    
                    mediaRecorder.start();
                    isRecording = true;
                    log('Started recording');
                    
                } catch (error) {
                    log(`Error starting recording: ${error}`);
                }
            }
            
            function stopConversation() {
                if (mediaRecorder && isRecording) {
                    mediaRecorder.stop();
                    isRecording = false;
                    log('Stopped recording');
                }
            }
            
            function playAudio(audioData) {
                const audio = new Audio('data:audio/mp3;base64,' + audioData);
                audio.play();
            }
            
            function clearLog() {
                document.getElementById('log').innerHTML = '';
            }
            
            async function updateStats() {
                try {
                    const response = await fetch('/api/stats');
                    if (response.ok) {
                        const stats = await response.json();
                        document.getElementById('stats').innerHTML = `
                            <p>Exchanges: ${stats.total_exchanges}</p>
                            <p>Avg Latency: ${stats.average_latency.toFixed(2)}s</p>
                            <p>Duration: ${stats.conversation_duration.toFixed(2)}s</p>
                        `;
                    }
                } catch (error) {
                    console.error('Error updating stats:', error);
                }
            }
            
            // Update stats every 5 seconds
            setInterval(updateStats, 5000);
        </script>
    </body>
    </html>
    """


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time voice interaction."""
    await websocket.accept()
    logger.info("WebSocket connection established")
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message["type"] == "audio":
                # Decode base64 audio
                audio_data = base64.b64decode(message["data"])
                
                # Process through voice agent
                if "default" in agents:
                    agent = agents["default"]
                    response_audio = agent.process_audio(audio_data)
                    
                    # Send response
                    response_data = base64.b64encode(response_audio).decode()
                    await websocket.send_text(json.dumps({
                        "type": "response",
                        "audio_data": response_data,
                        "response": "Audio processed successfully"
                    }))
                else:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": "No agent initialized"
                    }))
                    
    except WebSocketDisconnect:
        logger.info("WebSocket connection closed")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")


@app.post("/api/agent/initialize")
async def initialize_agent(config: AgentConfig):
    """Initialize a voice agent."""
    try:
        agent = create_agent(config.agent_type, config.config)
        agents["default"] = agent
        return {"status": "success", "message": f"Agent {config.agent_type} initialized"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    if "default" not in agents:
        return HealthResponse(
            status="unhealthy",
            components={"agent": "not_initialized"},
            timestamp=""
        )
    
    health = agents["default"].health_check()
    return HealthResponse(**health)


@app.get("/api/stats")
async def get_stats():
    """Get performance statistics."""
    if "default" not in agents:
        return {"error": "No agent initialized"}
    
    return agents["default"].get_performance_stats()


@app.post("/api/audio")
async def process_audio(request: AudioRequest):
    """Process audio data via REST API."""
    if "default" not in agents:
        raise HTTPException(status_code=400, detail="No agent initialized")
    
    try:
        # Decode base64 audio
        audio_data = base64.b64decode(request.audio_data)
        
        # Process through agent
        agent = agents["default"]
        response_audio = agent.process_audio(audio_data)
        
        # Encode response
        response_data = base64.b64encode(response_audio).decode()
        
        return {
            "status": "success",
            "audio_data": response_data,
            "session_id": request.session_id
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/conversation/start")
async def start_conversation():
    """Start a new conversation."""
    if "default" not in agents:
        raise HTTPException(status_code=400, detail="No agent initialized")
    
    agents["default"].start_conversation()
    return {"status": "success", "message": "Conversation started"}


@app.post("/api/conversation/stop")
async def stop_conversation():
    """Stop the current conversation."""
    if "default" not in agents:
        raise HTTPException(status_code=400, detail="No agent initialized")
    
    agents["default"].stop_conversation()
    return {"status": "success", "message": "Conversation stopped"}


@app.post("/api/conversation/reset")
async def reset_conversation():
    """Reset the conversation state."""
    if "default" not in agents:
        raise HTTPException(status_code=400, detail="No agent initialized")
    
    agents["default"].reset_conversation()
    return {"status": "success", "message": "Conversation reset"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
