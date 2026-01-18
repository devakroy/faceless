"""
Configuration loader utility for the faceless YouTube automation system.
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class AIConfig:
    provider: str = "ollama"
    model: str = "llama3.2"
    groq_api_key: str = ""
    huggingface_token: str = ""
    temperature: float = 0.8
    max_tokens: int = 1000


@dataclass
class TTSConfig:
    provider: str = "piper"
    voice: str = "en_US-lessac-medium"
    speed: float = 1.0
    pitch: float = 1.0


@dataclass
class VideoConfig:
    resolution: str = "1080x1920"
    fps: int = 30
    use_gpu: bool = True
    background_type: str = "stock_video"


@dataclass
class MediaConfig:
    pexels_api_key: str = ""
    pixabay_api_key: str = ""
    preferred_source: str = "pexels"


@dataclass
class SubtitleConfig:
    enabled: bool = True
    style: str = "modern"
    font: str = "Montserrat-Bold"
    font_size: int = 60
    color: str = "#FFFFFF"
    stroke_color: str = "#000000"
    stroke_width: int = 3
    position: str = "center"
    word_highlight: bool = True
    highlight_color: str = "#FFD700"


@dataclass
class ThumbnailConfig:
    enabled: bool = True
    style: str = "bold_text"
    use_ai: bool = False


@dataclass
class AIVideoConfig:
    provider: str = "diffusers"
    video_model: str = "stabilityai/stable-video-diffusion-img2vid-xt"
    image_model: str = "stabilityai/sdxl-turbo"
    width: int = 576
    height: int = 1024
    fps: int = 8
    clip_duration: float = 4.0
    max_frames: int = 25
    num_inference_steps: int = 8
    guidance_scale: float = 1.0
    seed: int = 42
    base_image_path: str = ""


@dataclass
class YouTubeConfig:
    client_secrets_file: str = "config/client_secrets.json"
    credentials_file: str = "config/youtube_credentials.json"
    privacy_status: str = "public"
    category_id: str = "22"
    default_tags: list = field(default_factory=lambda: ["shorts", "viral", "trending"])
    schedule_enabled: bool = True
    upload_times: list = field(default_factory=lambda: ["09:00", "15:00", "21:00"])


@dataclass
class ChannelConfig:
    name: str = "Your Channel Name"
    niche: str = "motivation"
    language: str = "en"
    upload_frequency: str = "daily"
    target_duration: int = 60
    video_format: str = "shorts"


@dataclass
class StorageConfig:
    output_dir: str = "output"
    temp_dir: str = "temp"
    archive_videos: bool = True
    max_storage_gb: int = 50


@dataclass
class Config:
    """Main configuration class that holds all settings."""
    channel: ChannelConfig = field(default_factory=ChannelConfig)
    ai: AIConfig = field(default_factory=AIConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    media: MediaConfig = field(default_factory=MediaConfig)
    subtitles: SubtitleConfig = field(default_factory=SubtitleConfig)
    thumbnail: ThumbnailConfig = field(default_factory=ThumbnailConfig)
    ai_video: AIVideoConfig = field(default_factory=AIVideoConfig)
    youtube: YouTubeConfig = field(default_factory=YouTubeConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)


class ConfigLoader:
    """Loads and manages configuration from YAML files."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = Path(config_path)
        self.config: Optional[Config] = None
        self._raw_config: Dict[str, Any] = {}
    
    def load(self) -> Config:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            self._raw_config = yaml.safe_load(f)
        
        self.config = self._parse_config(self._raw_config)
        return self.config
    
    def _parse_config(self, raw: Dict[str, Any]) -> Config:
        """Parse raw YAML config into Config dataclass."""
        config = Config()
        
        # Parse channel config
        if 'channel' in raw:
            config.channel = ChannelConfig(**raw['channel'])
        
        # Parse AI config
        if 'ai' in raw:
            config.ai = AIConfig(**raw['ai'])
        
        # Parse TTS config
        if 'tts' in raw:
            config.tts = TTSConfig(**raw['tts'])
        
        # Parse video config
        if 'video' in raw:
            config.video = VideoConfig(**raw['video'])
        
        # Parse media config
        if 'media' in raw:
            config.media = MediaConfig(**raw['media'])
        
        # Parse subtitle config
        if 'subtitles' in raw:
            config.subtitles = SubtitleConfig(**raw['subtitles'])
        
        # Parse thumbnail config
        if 'thumbnail' in raw:
            config.thumbnail = ThumbnailConfig(**raw['thumbnail'])

        # Parse AI video config
        if 'ai_video' in raw:
            config.ai_video = AIVideoConfig(**raw['ai_video'])
        
        # Parse YouTube config
        if 'youtube' in raw:
            yt_config = raw['youtube'].copy()
            config.youtube = YouTubeConfig(**yt_config)
        
        # Parse storage config
        if 'storage' in raw:
            config.storage = StorageConfig(**raw['storage'])
        
        return config
    
    def get_raw(self, key: str, default: Any = None) -> Any:
        """Get raw config value by dot-notation key."""
        keys = key.split('.')
        value = self._raw_config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    def ensure_directories(self):
        """Create necessary directories for the application."""
        if self.config is None:
            self.load()
        
        directories = [
            self.config.storage.output_dir,
            self.config.storage.temp_dir,
            "logs",
            "config",
            "assets/fonts",
            "assets/music",
            "assets/images",
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)


# Global config instance
_config_loader: Optional[ConfigLoader] = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _config_loader
    if _config_loader is None:
        _config_loader = ConfigLoader()
        _config_loader.load()
        _config_loader.ensure_directories()
    return _config_loader.config


def reload_config() -> Config:
    """Reload configuration from file."""
    global _config_loader
    _config_loader = ConfigLoader()
    _config_loader.load()
    _config_loader.ensure_directories()
    return _config_loader.config
