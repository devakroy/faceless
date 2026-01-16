"""
Stock Media Sourcing for Faceless YouTube Videos
Supports FREE stock media APIs:
- Pexels (200 requests/hour, free)
- Pixabay (100 requests/min, free)
- Local media library
"""

import os
import random
import hashlib
import asyncio
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from urllib.parse import urljoin
import logging
import requests
import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class MediaItem:
    """Represents a media item (video or image)."""
    id: str
    url: str
    download_url: str
    media_type: str  # "video" or "image"
    width: int
    height: int
    duration: Optional[float] = None  # For videos
    source: str = ""  # pexels, pixabay, local
    local_path: Optional[str] = None
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class MediaProvider(ABC):
    """Abstract base class for media providers."""
    
    @abstractmethod
    def search_videos(self, query: str, count: int = 5, orientation: str = "portrait") -> List[MediaItem]:
        """Search for videos."""
        pass
    
    @abstractmethod
    def search_images(self, query: str, count: int = 5, orientation: str = "portrait") -> List[MediaItem]:
        """Search for images."""
        pass
    
    @abstractmethod
    def download(self, item: MediaItem, output_dir: str) -> str:
        """Download media item and return local path."""
        pass


class PexelsProvider(MediaProvider):
    """
    Pexels API - FREE stock videos and images.
    Rate limit: 200 requests/hour, 20,000/month
    
    Get API key: https://www.pexels.com/api/
    """
    
    BASE_URL = "https://api.pexels.com"
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {"Authorization": api_key}
        self.cache_dir = Path("cache/pexels")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def search_videos(self, query: str, count: int = 5, orientation: str = "portrait") -> List[MediaItem]:
        """Search for videos on Pexels."""
        url = f"{self.BASE_URL}/videos/search"
        params = {
            "query": query,
            "per_page": min(count, 80),
            "orientation": orientation,
            "size": "medium"
        }
        
        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            items = []
            for video in data.get("videos", []):
                # Get the best quality video file
                video_files = video.get("video_files", [])
                best_file = self._get_best_video_file(video_files, orientation)
                
                if best_file:
                    items.append(MediaItem(
                        id=str(video["id"]),
                        url=video["url"],
                        download_url=best_file["link"],
                        media_type="video",
                        width=best_file.get("width", 1080),
                        height=best_file.get("height", 1920),
                        duration=video.get("duration", 0),
                        source="pexels",
                        tags=[]
                    ))
            
            return items[:count]
        
        except Exception as e:
            logger.error(f"Pexels video search failed: {e}")
            return []
    
    def _get_best_video_file(self, video_files: List[Dict], orientation: str) -> Optional[Dict]:
        """Get the best quality video file for the orientation."""
        target_height = 1920 if orientation == "portrait" else 1080
        
        # Sort by quality (height)
        sorted_files = sorted(
            video_files,
            key=lambda x: abs(x.get("height", 0) - target_height)
        )
        
        # Prefer HD quality
        for f in sorted_files:
            if f.get("quality") in ["hd", "sd"] and f.get("height", 0) >= 720:
                return f
        
        return sorted_files[0] if sorted_files else None
    
    def search_images(self, query: str, count: int = 5, orientation: str = "portrait") -> List[MediaItem]:
        """Search for images on Pexels."""
        url = f"{self.BASE_URL}/v1/search"
        params = {
            "query": query,
            "per_page": min(count, 80),
            "orientation": orientation
        }
        
        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            items = []
            for photo in data.get("photos", []):
                items.append(MediaItem(
                    id=str(photo["id"]),
                    url=photo["url"],
                    download_url=photo["src"]["large2x"],
                    media_type="image",
                    width=photo["width"],
                    height=photo["height"],
                    source="pexels",
                    tags=[]
                ))
            
            return items[:count]
        
        except Exception as e:
            logger.error(f"Pexels image search failed: {e}")
            return []
    
    def download(self, item: MediaItem, output_dir: str) -> str:
        """Download media from Pexels."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        ext = "mp4" if item.media_type == "video" else "jpg"
        filename = f"pexels_{item.id}.{ext}"
        filepath = output_path / filename
        
        # Check cache
        if filepath.exists():
            logger.info(f"Using cached: {filepath}")
            return str(filepath)
        
        # Download
        logger.info(f"Downloading from Pexels: {item.id}")
        response = requests.get(item.download_url, stream=True, timeout=120)
        response.raise_for_status()
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        item.local_path = str(filepath)
        return str(filepath)


class PixabayProvider(MediaProvider):
    """
    Pixabay API - FREE stock videos and images.
    Rate limit: 100 requests/minute
    
    Get API key: https://pixabay.com/api/docs/
    """
    
    BASE_URL = "https://pixabay.com/api"
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.cache_dir = Path("cache/pixabay")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def search_videos(self, query: str, count: int = 5, orientation: str = "portrait") -> List[MediaItem]:
        """Search for videos on Pixabay."""
        url = f"{self.BASE_URL}/videos/"
        
        # Map orientation
        video_type = "all"
        
        params = {
            "key": self.api_key,
            "q": query,
            "per_page": min(count, 200),
            "video_type": video_type,
            "safesearch": "true"
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            items = []
            for video in data.get("hits", []):
                # Get medium quality video
                videos = video.get("videos", {})
                medium = videos.get("medium", videos.get("small", {}))
                
                if medium:
                    items.append(MediaItem(
                        id=str(video["id"]),
                        url=video["pageURL"],
                        download_url=medium.get("url", ""),
                        media_type="video",
                        width=medium.get("width", 1080),
                        height=medium.get("height", 1920),
                        duration=video.get("duration", 0),
                        source="pixabay",
                        tags=video.get("tags", "").split(", ")
                    ))
            
            return items[:count]
        
        except Exception as e:
            logger.error(f"Pixabay video search failed: {e}")
            return []
    
    def search_images(self, query: str, count: int = 5, orientation: str = "portrait") -> List[MediaItem]:
        """Search for images on Pixabay."""
        url = f"{self.BASE_URL}/"
        
        params = {
            "key": self.api_key,
            "q": query,
            "per_page": min(count, 200),
            "orientation": "vertical" if orientation == "portrait" else "horizontal",
            "safesearch": "true",
            "image_type": "photo"
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            items = []
            for image in data.get("hits", []):
                items.append(MediaItem(
                    id=str(image["id"]),
                    url=image["pageURL"],
                    download_url=image["largeImageURL"],
                    media_type="image",
                    width=image["imageWidth"],
                    height=image["imageHeight"],
                    source="pixabay",
                    tags=image.get("tags", "").split(", ")
                ))
            
            return items[:count]
        
        except Exception as e:
            logger.error(f"Pixabay image search failed: {e}")
            return []
    
    def download(self, item: MediaItem, output_dir: str) -> str:
        """Download media from Pixabay."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        ext = "mp4" if item.media_type == "video" else "jpg"
        filename = f"pixabay_{item.id}.{ext}"
        filepath = output_path / filename
        
        # Check cache
        if filepath.exists():
            logger.info(f"Using cached: {filepath}")
            return str(filepath)
        
        # Download
        logger.info(f"Downloading from Pixabay: {item.id}")
        response = requests.get(item.download_url, stream=True, timeout=120)
        response.raise_for_status()
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        item.local_path = str(filepath)
        return str(filepath)


class LocalMediaProvider(MediaProvider):
    """
    Local media library provider.
    Uses pre-downloaded or user-provided media.
    """
    
    def __init__(self, media_dir: str = "assets/media"):
        self.media_dir = Path(media_dir)
        self.media_dir.mkdir(parents=True, exist_ok=True)
        self._index_media()
    
    def _index_media(self):
        """Index local media files."""
        self.videos = []
        self.images = []
        
        video_extensions = {'.mp4', '.mov', '.avi', '.mkv', '.webm'}
        image_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
        
        for file in self.media_dir.rglob('*'):
            if file.is_file():
                ext = file.suffix.lower()
                if ext in video_extensions:
                    self.videos.append(file)
                elif ext in image_extensions:
                    self.images.append(file)
        
        logger.info(f"Indexed {len(self.videos)} videos and {len(self.images)} images")
    
    def search_videos(self, query: str, count: int = 5, orientation: str = "portrait") -> List[MediaItem]:
        """Search local videos (returns random selection)."""
        if not self.videos:
            return []
        
        selected = random.sample(self.videos, min(count, len(self.videos)))
        
        items = []
        for video_path in selected:
            items.append(MediaItem(
                id=hashlib.md5(str(video_path).encode()).hexdigest()[:8],
                url=str(video_path),
                download_url=str(video_path),
                media_type="video",
                width=1080,
                height=1920,
                source="local",
                local_path=str(video_path)
            ))
        
        return items
    
    def search_images(self, query: str, count: int = 5, orientation: str = "portrait") -> List[MediaItem]:
        """Search local images (returns random selection)."""
        if not self.images:
            return []
        
        selected = random.sample(self.images, min(count, len(self.images)))
        
        items = []
        for image_path in selected:
            items.append(MediaItem(
                id=hashlib.md5(str(image_path).encode()).hexdigest()[:8],
                url=str(image_path),
                download_url=str(image_path),
                media_type="image",
                width=1080,
                height=1920,
                source="local",
                local_path=str(image_path)
            ))
        
        return items
    
    def download(self, item: MediaItem, output_dir: str) -> str:
        """For local media, just return the path."""
        return item.local_path or item.download_url


class StockMediaManager:
    """
    Manages stock media sourcing from multiple providers.
    Handles caching, fallbacks, and smart selection.
    """
    
    # Keywords for different niches
    NICHE_KEYWORDS = {
        "motivation": [
            "success", "mountain top", "sunrise", "running", "workout",
            "city skyline", "businessman", "achievement", "nature", "ocean"
        ],
        "facts": [
            "science", "space", "nature", "technology", "abstract",
            "universe", "brain", "laboratory", "earth", "microscope"
        ],
        "stories": [
            "dramatic", "cinematic", "dark", "mystery", "urban",
            "night city", "rain", "fog", "silhouette", "emotional"
        ],
        "tech": [
            "technology", "computer", "coding", "futuristic", "digital",
            "smartphone", "circuit", "robot", "ai", "data"
        ],
        "finance": [
            "money", "business", "stock market", "coins", "wealth",
            "office", "charts", "investment", "bank", "gold"
        ],
        "health": [
            "fitness", "healthy food", "meditation", "yoga", "running",
            "gym", "nature", "wellness", "sleep", "vegetables"
        ],
        "mystery": [
            "dark", "mysterious", "fog", "abandoned", "night",
            "shadows", "creepy", "forest", "old building", "paranormal"
        ]
    }
    
    def __init__(self, config):
        self.config = config
        self.providers: List[MediaProvider] = []
        self.cache_dir = Path("cache/media")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self._init_providers()
    
    def _init_providers(self):
        """Initialize available providers."""
        # Add Pexels if API key available
        if self.config.media.pexels_api_key:
            self.providers.append(PexelsProvider(self.config.media.pexels_api_key))
            logger.info("Pexels provider initialized")
        
        # Add Pixabay if API key available
        if self.config.media.pixabay_api_key:
            self.providers.append(PixabayProvider(self.config.media.pixabay_api_key))
            logger.info("Pixabay provider initialized")
        
        # Always add local provider as fallback
        self.providers.append(LocalMediaProvider())
        logger.info("Local media provider initialized")
    
    def get_background_videos(self, niche: str, count: int = 3, 
                              duration_needed: float = 60) -> List[str]:
        """
        Get background videos for a specific niche.
        Returns list of local file paths.
        """
        keywords = self.NICHE_KEYWORDS.get(niche, self.NICHE_KEYWORDS["motivation"])
        query = random.choice(keywords)
        
        orientation = "portrait" if "1920" in self.config.video.resolution else "landscape"
        
        all_items = []
        
        # Try each provider
        for provider in self.providers:
            try:
                items = provider.search_videos(query, count * 2, orientation)
                all_items.extend(items)
                
                if len(all_items) >= count:
                    break
            except Exception as e:
                logger.warning(f"Provider failed: {e}")
                continue
        
        # Download videos
        downloaded = []
        for item in all_items[:count]:
            try:
                provider = self._get_provider_for_item(item)
                local_path = provider.download(item, str(self.cache_dir / "videos"))
                downloaded.append(local_path)
            except Exception as e:
                logger.error(f"Failed to download {item.id}: {e}")
        
        return downloaded
    
    def get_background_images(self, niche: str, count: int = 5) -> List[str]:
        """
        Get background images for a specific niche.
        Returns list of local file paths.
        """
        keywords = self.NICHE_KEYWORDS.get(niche, self.NICHE_KEYWORDS["motivation"])
        query = random.choice(keywords)
        
        orientation = "portrait" if "1920" in self.config.video.resolution else "landscape"
        
        all_items = []
        
        # Try each provider
        for provider in self.providers:
            try:
                items = provider.search_images(query, count * 2, orientation)
                all_items.extend(items)
                
                if len(all_items) >= count:
                    break
            except Exception as e:
                logger.warning(f"Provider failed: {e}")
                continue
        
        # Download images
        downloaded = []
        for item in all_items[:count]:
            try:
                provider = self._get_provider_for_item(item)
                local_path = provider.download(item, str(self.cache_dir / "images"))
                downloaded.append(local_path)
            except Exception as e:
                logger.error(f"Failed to download {item.id}: {e}")
        
        return downloaded
    
    def _get_provider_for_item(self, item: MediaItem) -> MediaProvider:
        """Get the appropriate provider for a media item."""
        for provider in self.providers:
            if isinstance(provider, PexelsProvider) and item.source == "pexels":
                return provider
            elif isinstance(provider, PixabayProvider) and item.source == "pixabay":
                return provider
            elif isinstance(provider, LocalMediaProvider) and item.source == "local":
                return provider
        
        return self.providers[-1]  # Fallback to local


def create_stock_media_manager(config) -> StockMediaManager:
    """Factory function to create stock media manager."""
    return StockMediaManager(config)
