"""
Text-to-Speech Engine for Faceless YouTube Videos
Supports multiple FREE TTS providers:
- Piper TTS (local, fast, high quality)
- Edge TTS (Microsoft's free TTS)
- Coqui TTS (local, neural voices)
"""

import os
import subprocess
import tempfile
import asyncio
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, List, Tuple
from dataclasses import dataclass
import logging
import wave
import json

logger = logging.getLogger(__name__)


@dataclass
class AudioResult:
    """Result of TTS generation."""
    audio_path: str
    duration: float
    sample_rate: int
    word_timestamps: Optional[List[Tuple[str, float, float]]] = None  # (word, start, end)


class TTSProvider(ABC):
    """Abstract base class for TTS providers."""
    
    @abstractmethod
    def synthesize(self, text: str, output_path: str) -> AudioResult:
        """Synthesize speech from text."""
        pass
    
    @abstractmethod
    async def synthesize_async(self, text: str, output_path: str) -> AudioResult:
        """Async version of synthesize."""
        pass
    
    def get_audio_duration(self, audio_path: str) -> float:
        """Get duration of an audio file."""
        try:
            with wave.open(audio_path, 'rb') as wav_file:
                frames = wav_file.getnframes()
                rate = wav_file.getframerate()
                return frames / float(rate)
        except Exception:
            # Fallback using ffprobe
            try:
                result = subprocess.run(
                    ['ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
                     '-of', 'default=noprint_wrappers=1:nokey=1', audio_path],
                    capture_output=True, text=True
                )
                return float(result.stdout.strip())
            except Exception as e:
                logger.error(f"Failed to get audio duration: {e}")
                return 0.0


class PiperTTS(TTSProvider):
    """
    Piper TTS - Fast, local, high-quality TTS.
    Completely FREE and runs on CPU.
    
    Install: pip install piper-tts
    Models: https://github.com/rhasspy/piper/releases
    """
    
    VOICE_MODELS = {
        "en_US-lessac-medium": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx",
        "en_US-amy-medium": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx",
        "en_US-ryan-medium": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/medium/en_US-ryan-medium.onnx",
        "en_GB-alan-medium": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/alan/medium/en_GB-alan-medium.onnx",
    }
    
    def __init__(self, voice: str = "en_US-lessac-medium", speed: float = 1.0):
        self.voice = voice
        self.speed = speed
        self.models_dir = Path("models/piper")
        self.models_dir.mkdir(parents=True, exist_ok=True)
    
    def _ensure_model(self) -> Path:
        """Download model if not present."""
        model_path = self.models_dir / f"{self.voice}.onnx"
        config_path = self.models_dir / f"{self.voice}.onnx.json"
        
        if not model_path.exists():
            logger.info(f"Downloading Piper voice model: {self.voice}")
            
            # Download model
            model_url = self.VOICE_MODELS.get(self.voice)
            if model_url:
                subprocess.run(['wget', '-q', '-O', str(model_path), model_url], check=True)
                subprocess.run(['wget', '-q', '-O', str(config_path), model_url + '.json'], check=True)
            else:
                # Try using piper to download
                subprocess.run(['piper', '--download-dir', str(self.models_dir), 
                              '--model', self.voice, '--update-voices'], check=True)
        
        return model_path
    
    def synthesize(self, text: str, output_path: str) -> AudioResult:
        """Synthesize speech using Piper TTS."""
        model_path = self._ensure_model()
        
        # Ensure output is WAV
        if not output_path.endswith('.wav'):
            output_path = output_path.rsplit('.', 1)[0] + '.wav'
        
        # Run Piper
        cmd = [
            'piper',
            '--model', str(model_path),
            '--output_file', output_path,
            '--length_scale', str(1.0 / self.speed)
        ]
        
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        stdout, stderr = process.communicate(input=text.encode('utf-8'))
        
        if process.returncode != 0:
            logger.error(f"Piper TTS error: {stderr.decode()}")
            raise Exception(f"Piper TTS failed: {stderr.decode()}")
        
        duration = self.get_audio_duration(output_path)
        
        return AudioResult(
            audio_path=output_path,
            duration=duration,
            sample_rate=22050
        )
    
    async def synthesize_async(self, text: str, output_path: str) -> AudioResult:
        """Async synthesis using Piper."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.synthesize, text, output_path)


class EdgeTTS(TTSProvider):
    """
    Microsoft Edge TTS - FREE, high-quality, many voices.
    Uses Microsoft's online TTS service (free, no API key needed).
    
    Install: pip install edge-tts
    """
    
    VOICES = {
        "en-US-GuyNeural": "Male, US English, Natural",
        "en-US-JennyNeural": "Female, US English, Natural",
        "en-US-AriaNeural": "Female, US English, Expressive",
        "en-US-DavisNeural": "Male, US English, Calm",
        "en-US-TonyNeural": "Male, US English, Friendly",
        "en-GB-RyanNeural": "Male, UK English",
        "en-GB-SoniaNeural": "Female, UK English",
        "en-AU-WilliamNeural": "Male, Australian English",
    }
    
    def __init__(self, voice: str = "en-US-GuyNeural", speed: float = 1.0, pitch: float = 1.0):
        self.voice = voice
        self.speed = speed
        self.pitch = pitch
    
    def _get_rate_string(self) -> str:
        """Convert speed to Edge TTS rate string."""
        percentage = int((self.speed - 1.0) * 100)
        if percentage >= 0:
            return f"+{percentage}%"
        return f"{percentage}%"
    
    def _get_pitch_string(self) -> str:
        """Convert pitch to Edge TTS pitch string."""
        hz = int((self.pitch - 1.0) * 50)
        if hz >= 0:
            return f"+{hz}Hz"
        return f"{hz}Hz"
    
    def synthesize(self, text: str, output_path: str) -> AudioResult:
        """Synthesize speech using Edge TTS."""
        import edge_tts
        
        # Ensure output is MP3 (Edge TTS default)
        if not output_path.endswith('.mp3'):
            mp3_path = output_path.rsplit('.', 1)[0] + '.mp3'
        else:
            mp3_path = output_path
        
        # Run Edge TTS
        communicate = edge_tts.Communicate(
            text,
            self.voice,
            rate=self._get_rate_string(),
            pitch=self._get_pitch_string()
        )
        
        # Run synchronously
        asyncio.get_event_loop().run_until_complete(
            communicate.save(mp3_path)
        )
        
        # Convert to WAV if needed
        wav_path = output_path.rsplit('.', 1)[0] + '.wav'
        subprocess.run([
            'ffmpeg', '-y', '-i', mp3_path,
            '-acodec', 'pcm_s16le', '-ar', '22050',
            wav_path
        ], capture_output=True)
        
        duration = self.get_audio_duration(wav_path)
        
        return AudioResult(
            audio_path=wav_path,
            duration=duration,
            sample_rate=22050
        )
    
    async def synthesize_async(self, text: str, output_path: str) -> AudioResult:
        """Async synthesis using Edge TTS."""
        import edge_tts
        
        # Ensure output paths
        mp3_path = output_path.rsplit('.', 1)[0] + '.mp3'
        wav_path = output_path.rsplit('.', 1)[0] + '.wav'
        
        # Run Edge TTS
        communicate = edge_tts.Communicate(
            text,
            self.voice,
            rate=self._get_rate_string(),
            pitch=self._get_pitch_string()
        )
        
        await communicate.save(mp3_path)
        
        # Convert to WAV
        process = await asyncio.create_subprocess_exec(
            'ffmpeg', '-y', '-i', mp3_path,
            '-acodec', 'pcm_s16le', '-ar', '22050',
            wav_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        await process.communicate()
        
        duration = self.get_audio_duration(wav_path)
        
        return AudioResult(
            audio_path=wav_path,
            duration=duration,
            sample_rate=22050
        )


class CoquiTTS(TTSProvider):
    """
    Coqui TTS - Open source, neural TTS.
    Runs locally on CPU/GPU, completely FREE.
    
    Install: pip install TTS
    """
    
    MODELS = {
        "tts_models/en/ljspeech/tacotron2-DDC": "Female, US English, High Quality",
        "tts_models/en/ljspeech/glow-tts": "Female, US English, Fast",
        "tts_models/en/vctk/vits": "Multi-speaker, UK English",
    }
    
    def __init__(self, model: str = "tts_models/en/ljspeech/tacotron2-DDC", 
                 use_gpu: bool = False, speed: float = 1.0):
        self.model_name = model
        self.use_gpu = use_gpu
        self.speed = speed
        self._tts = None
    
    def _get_tts(self):
        """Lazy load TTS model."""
        if self._tts is None:
            from TTS.api import TTS
            self._tts = TTS(model_name=self.model_name, gpu=self.use_gpu)
        return self._tts
    
    def synthesize(self, text: str, output_path: str) -> AudioResult:
        """Synthesize speech using Coqui TTS."""
        tts = self._get_tts()
        
        # Ensure output is WAV
        wav_path = output_path.rsplit('.', 1)[0] + '.wav'
        
        # Generate speech
        tts.tts_to_file(
            text=text,
            file_path=wav_path,
            speed=self.speed
        )
        
        duration = self.get_audio_duration(wav_path)
        
        return AudioResult(
            audio_path=wav_path,
            duration=duration,
            sample_rate=22050
        )
    
    async def synthesize_async(self, text: str, output_path: str) -> AudioResult:
        """Async synthesis using Coqui TTS."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.synthesize, text, output_path)


class TTSEngine:
    """Main TTS engine that manages providers and generates audio."""
    
    def __init__(self, provider: TTSProvider):
        self.provider = provider
        self.temp_dir = Path("temp/audio")
        self.temp_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_speech(self, text: str, output_path: Optional[str] = None) -> AudioResult:
        """Generate speech from text."""
        if output_path is None:
            output_path = str(self.temp_dir / f"speech_{hash(text)}.wav")
        
        logger.info(f"Generating speech: {text[:50]}...")
        
        result = self.provider.synthesize(text, output_path)
        
        logger.info(f"Generated audio: {result.duration:.2f}s")
        
        return result
    
    async def generate_speech_async(self, text: str, output_path: Optional[str] = None) -> AudioResult:
        """Async version of generate_speech."""
        if output_path is None:
            output_path = str(self.temp_dir / f"speech_{hash(text)}.wav")
        
        logger.info(f"Generating speech: {text[:50]}...")
        
        result = await self.provider.synthesize_async(text, output_path)
        
        logger.info(f"Generated audio: {result.duration:.2f}s")
        
        return result
    
    def generate_with_timestamps(self, text: str, output_path: Optional[str] = None) -> AudioResult:
        """Generate speech and extract word timestamps using Whisper."""
        result = self.generate_speech(text, output_path)
        
        # Use Whisper to get word-level timestamps
        try:
            timestamps = self._extract_timestamps(result.audio_path)
            result.word_timestamps = timestamps
        except Exception as e:
            logger.warning(f"Failed to extract timestamps: {e}")
        
        return result
    
    def _extract_timestamps(self, audio_path: str) -> List[Tuple[str, float, float]]:
        """Extract word timestamps using Whisper."""
        try:
            import whisper
            
            model = whisper.load_model("tiny")  # Use tiny for speed
            result = model.transcribe(audio_path, word_timestamps=True)
            
            timestamps = []
            for segment in result.get("segments", []):
                for word_info in segment.get("words", []):
                    timestamps.append((
                        word_info["word"].strip(),
                        word_info["start"],
                        word_info["end"]
                    ))
            
            return timestamps
        except ImportError:
            logger.warning("Whisper not installed, skipping timestamp extraction")
            return []


def create_tts_engine(config) -> TTSEngine:
    """Factory function to create TTS engine based on config."""
    provider_name = config.tts.provider.lower()
    
    if provider_name == "piper":
        provider = PiperTTS(
            voice=config.tts.voice,
            speed=config.tts.speed
        )
    elif provider_name == "edge_tts" or provider_name == "edge":
        provider = EdgeTTS(
            voice=config.tts.voice if hasattr(config.tts, 'voice') else "en-US-GuyNeural",
            speed=config.tts.speed,
            pitch=config.tts.pitch
        )
    elif provider_name == "coqui":
        provider = CoquiTTS(
            model=config.tts.voice if hasattr(config.tts, 'voice') else "tts_models/en/ljspeech/tacotron2-DDC",
            use_gpu=config.video.use_gpu if hasattr(config.video, 'use_gpu') else False,
            speed=config.tts.speed
        )
    else:
        # Default to Edge TTS (most reliable, no setup needed)
        logger.warning(f"Unknown TTS provider: {provider_name}, using Edge TTS")
        provider = EdgeTTS()
    
    return TTSEngine(provider)
