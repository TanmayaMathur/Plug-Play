"""
Command-line interface for the Voice Agent Framework.
"""

import argparse
import sys
import time
import pyaudio
import wave
import threading
import queue
from typing import Optional

from .core import VoiceAgent
from .agents import BasicAgent, HealthcareAgent, CustomerSupportAgent
from .config import VoiceAgentConfig
from loguru import logger


class AudioRecorder:
    """Simple audio recorder for CLI interface."""
    
    def __init__(self, sample_rate=16000, chunk_size=1024):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.audio = pyaudio.PyAudio()
        self.frames = []
        self.is_recording = False
        
    def start_recording(self):
        """Start recording audio."""
        self.frames = []
        self.is_recording = True
        
        def callback(in_data, frame_count, time_info, status):
            if self.is_recording:
                self.frames.append(in_data)
            return (in_data, pyaudio.paContinue)
        
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=callback
        )
        
        self.stream.start_stream()
        logger.info("Recording started...")
    
    def stop_recording(self) -> bytes:
        """Stop recording and return audio data."""
        self.is_recording = False
        
        if hasattr(self, 'stream'):
            self.stream.stop_stream()
            self.stream.close()
        
        # Convert frames to bytes
        audio_data = b''.join(self.frames)
        logger.info("Recording stopped.")
        return audio_data
    
    def close(self):
        """Close the audio interface."""
        self.audio.terminate()


class AudioPlayer:
    """Simple audio player for CLI interface."""
    
    def __init__(self, sample_rate=22050):
        self.sample_rate = sample_rate
        self.audio = pyaudio.PyAudio()
    
    def play_audio(self, audio_data: bytes):
        """Play audio data."""
        try:
            stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                output=True
            )
            
            stream.write(audio_data)
            stream.stop_stream()
            stream.close()
            
        except Exception as e:
            logger.error(f"Error playing audio: {e}")
    
    def close(self):
        """Close the audio interface."""
        self.audio.terminate()


def create_agent_from_args(args) -> VoiceAgent:
    """Create a voice agent based on command line arguments."""
    agent_map = {
        "basic": BasicAgent,
        "healthcare": HealthcareAgent,
        "customer_support": CustomerSupportAgent
    }
    
    if args.agent not in agent_map:
        logger.error(f"Unknown agent type: {args.agent}")
        sys.exit(1)
    
    agent_class = agent_map[args.agent]
    agent_instance = agent_class()
    
    config = {
        "stt_engine": args.stt_engine,
        "tts_engine": args.tts_engine,
        "domain": args.agent
    }
    
    return VoiceAgent(
        stt_engine=args.stt_engine,
        llm_agent=agent_instance,
        tts_engine=args.tts_engine,
        config=config
    )


def interactive_mode(agent: VoiceAgent):
    """Run interactive voice conversation mode."""
    print("\n🎤 Voice Agent Interactive Mode")
    print("=" * 40)
    print("Commands:")
    print("  'start' - Start recording")
    print("  'stop'  - Stop recording and process")
    print("  'quit'  - Exit the program")
    print("  'stats' - Show performance statistics")
    print("  'help'  - Show this help")
    print("=" * 40)
    
    recorder = AudioRecorder()
    player = AudioPlayer()
    
    try:
        agent.start_conversation()
        
        while True:
            command = input("\n> ").strip().lower()
            
            if command == "quit":
                break
            elif command == "help":
                print("Commands: start, stop, quit, stats, help")
            elif command == "stats":
                stats = agent.get_performance_stats()
                print(f"\nPerformance Statistics:")
                print(f"  Total exchanges: {stats['total_exchanges']}")
                print(f"  Average latency: {stats['average_latency']:.2f}s")
                print(f"  Max latency: {stats['max_latency']:.2f}s")
                print(f"  Min latency: {stats['min_latency']:.2f}s")
                print(f"  Conversation duration: {stats['conversation_duration']:.2f}s")
            elif command == "start":
                print("Recording... (say 'stop' to stop recording)")
                recorder.start_recording()
            elif command == "stop":
                if recorder.is_recording:
                    audio_data = recorder.stop_recording()
                    
                    if audio_data:
                        print("Processing audio...")
                        start_time = time.time()
                        
                        response_audio = agent.process_audio(audio_data)
                        
                        if response_audio:
                            latency = time.time() - start_time
                            print(f"Response generated in {latency:.2f}s")
                            print("Playing response...")
                            player.play_audio(response_audio)
                        else:
                            print("No response generated.")
                else:
                    print("Not currently recording.")
            else:
                print("Unknown command. Type 'help' for available commands.")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        agent.stop_conversation()
        recorder.close()
        player.close()
        print("Voice agent session ended.")


def text_mode(agent: VoiceAgent):
    """Run text-based conversation mode."""
    print("\n💬 Voice Agent Text Mode")
    print("=" * 40)
    print("Type your messages and press Enter.")
    print("Type 'quit' to exit.")
    print("=" * 40)
    
    try:
        agent.start_conversation()
        
        while True:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() == "quit":
                break
            
            if user_input:
                print("Processing...")
                start_time = time.time()
                
                # Create a simple audio-like context for text processing
                context = agent.llm_agent.session_context
                response = agent.llm_agent.process(user_input, context)
                
                latency = time.time() - start_time
                print(f"Agent ({latency:.2f}s): {response}")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        agent.stop_conversation()
        print("Text session ended.")


def demo_mode(agent: VoiceAgent):
    """Run demo mode with predefined interactions."""
    print("\n🎭 Voice Agent Demo Mode")
    print("=" * 40)
    
    # Demo interactions based on agent type
    demos = {
        "basic": [
            "Hello, how are you?",
            "What's the weather like today?",
            "Tell me a joke",
            "What time is it?"
        ],
        "healthcare": [
            "I have a headache",
            "What are the symptoms of the flu?",
            "I need to schedule an appointment",
            "What should I do for a fever?"
        ],
        "customer_support": [
            "I want to return a product",
            "What's your shipping policy?",
            "My order is delayed",
            "How do I contact support?"
        ]
    }
    
    agent_type = type(agent.llm_agent).__name__.lower().replace("agent", "")
    interactions = demos.get(agent_type, demos["basic"])
    
    try:
        agent.start_conversation()
        
        for i, interaction in enumerate(interactions, 1):
            print(f"\n--- Demo {i}/{len(interactions)} ---")
            print(f"User: {interaction}")
            
            start_time = time.time()
            context = agent.llm_agent.session_context
            response = agent.llm_agent.process(interaction, context)
            latency = time.time() - start_time
            
            print(f"Agent ({latency:.2f}s): {response}")
            
            if i < len(interactions):
                input("\nPress Enter to continue...")
    
    except KeyboardInterrupt:
        print("\nDemo interrupted.")
    finally:
        agent.stop_conversation()
        print("Demo completed.")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Voice Agent Framework CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --agent basic --mode interactive
  %(prog)s --agent healthcare --mode text
  %(prog)s --agent customer_support --mode demo
        """
    )
    
    parser.add_argument(
        "--agent",
        choices=["basic", "healthcare", "customer_support"],
        default="basic",
        help="Type of agent to use (default: basic)"
    )
    
    parser.add_argument(
        "--mode",
        choices=["interactive", "text", "demo"],
        default="interactive",
        help="Interaction mode (default: interactive)"
    )
    
    parser.add_argument(
        "--stt-engine",
        choices=["whisper", "vosk"],
        default="whisper",
        help="Speech-to-Text engine (default: whisper)"
    )
    
    parser.add_argument(
        "--tts-engine",
        choices=["elevenlabs", "pyttsx3"],
        default="elevenlabs",
        help="Text-to-Speech engine (default: elevenlabs)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logger.add(sys.stderr, level="DEBUG")
    else:
        logger.add(sys.stderr, level="INFO")
    
    try:
        # Create agent
        print(f"Initializing {args.agent} agent...")
        agent = create_agent_from_args(args)
        
        # Health check
        health = agent.health_check()
        if health["overall"] != "healthy":
            print("Warning: Some components are unhealthy:")
            for component, status in health["components"].items():
                print(f"  {component}: {status}")
        
        # Run selected mode
        if args.mode == "interactive":
            interactive_mode(agent)
        elif args.mode == "text":
            text_mode(agent)
        elif args.mode == "demo":
            demo_mode(agent)
    
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
