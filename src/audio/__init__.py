"""Audio modules for TTS and audio processing."""

from .tts_engine import (
    TTSEngine,
    TTSProvider,
    PiperTTS,
    EdgeTTS,
    CoquiTTS,
    AudioResult,
    create_tts_engine
)

__all__ = [
    'TTSEngine',
    'TTSProvider',
    'PiperTTS',
    'EdgeTTS',
    'CoquiTTS',
    'AudioResult',
    'create_tts_engine'
]
