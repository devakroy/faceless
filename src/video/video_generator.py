"""
Video Generator for Faceless YouTube Videos
Uses MoviePy and FFmpeg for CPU/GPU video rendering.
Includes subtitle generation, effects, and composition.
"""

import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass
import logging
import json
import math

# Updated for MoviePy 2.x where classes are directly in moviepy module
from moviepy import (
    VideoFileClip, AudioFileClip, ImageClip, TextClip,
    CompositeVideoClip, CompositeAudioClip, concatenate_videoclips,
    ColorClip, vfx
)
from moviepy.video.tools.subtitles import SubtitlesClip
from PIL import Image, ImageDraw, ImageFont
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SubtitleStyle:
    """Subtitle styling configuration."""
    font: str = "Montserrat-Bold"
    font_size: int = 60
    color: str = "#FFFFFF"
    stroke_color: str = "#000000"
    stroke_width: int = 3
    bg_color: Optional[str] = None
    position: str = "center"  # top, center, bottom
    word_highlight: bool = True
    highlight_color: str = "#FFD700"
    animation: str = "fade"  # fade, pop, slide


@dataclass
class VideoSpec:
    """Video specification."""
    width: int = 1080
    height: int = 1920
    fps: int = 30
    duration: float = 60.0
    format: str = "shorts"  # shorts or long


class SubtitleGenerator:
    """Generates animated subtitles for videos."""
    
    def __init__(self, style: SubtitleStyle):
        self.style = style
        self.fonts_dir = Path("assets/fonts")
        self.fonts_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_font_path(self) -> str:
        """Get path to font file, download if needed."""
        font_path = self.fonts_dir / f"{self.style.font}.ttf"
        
        if not font_path.exists():
            # Try system fonts
            system_fonts = [
                f"/usr/share/fonts/truetype/{self.style.font}.ttf",
                f"/usr/share/fonts/{self.style.font}.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            ]
            
            for sf in system_fonts:
                if os.path.exists(sf):
                    return sf
            
            # Fallback to default
            return "DejaVu-Sans-Bold"
        
        return str(font_path)
    
    def create_word_clips(self, words: List[Tuple[str, float, float]], 
                          video_size: Tuple[int, int]) -> List[TextClip]:
        """Create individual word clips with timing."""
        clips = []
        font_path = self._get_font_path()
        
        for word, start, end in words:
            try:
                clip = TextClip(
                    word,
                    fontsize=self.style.font_size,
                    font=font_path,
                    color=self.style.color,
                    stroke_color=self.style.stroke_color,
                    stroke_width=self.style.stroke_width,
                    method='caption',
                    size=(video_size[0] - 100, None)
                )
                
                clip = clip.set_start(start).set_end(end)
                clip = clip.set_position(('center', self._get_y_position(video_size[1])))
                
                # Add fade effect
                if self.style.animation == "fade":
                    clip = clip.crossfadein(0.1).crossfadeout(0.1)
                
                clips.append(clip)
            except Exception as e:
                logger.warning(f"Failed to create word clip for '{word}': {e}")
        
        return clips
    
    def create_sentence_clips(self, text: str, duration: float,
                              video_size: Tuple[int, int]) -> List[TextClip]:
        """Create sentence-based subtitle clips."""
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        clips = []
        time_per_sentence = duration / len(sentences) if sentences else duration
        current_time = 0
        
        font_path = self._get_font_path()
        
        for sentence in sentences:
            if not sentence.strip():
                continue
            
            try:
                clip = TextClip(
                    sentence.strip(),
                    fontsize=self.style.font_size,
                    font=font_path,
                    color=self.style.color,
                    stroke_color=self.style.stroke_color,
                    stroke_width=self.style.stroke_width,
                    method='caption',
                    size=(video_size[0] - 100, None),
                    align='center'
                )
                
                clip = clip.set_start(current_time).set_duration(time_per_sentence)
                clip = clip.set_position(('center', self._get_y_position(video_size[1])))
                
                # Add animation
                if self.style.animation == "fade":
                    clip = clip.crossfadein(0.2).crossfadeout(0.2)
                
                clips.append(clip)
                current_time += time_per_sentence
                
            except Exception as e:
                logger.warning(f"Failed to create sentence clip: {e}")
        
        return clips
    
    def _get_y_position(self, video_height: int) -> int:
        """Get Y position based on style setting."""
        if self.style.position == "top":
            return int(video_height * 0.15)
        elif self.style.position == "bottom":
            return int(video_height * 0.75)
        else:  # center
            return int(video_height * 0.45)


class VideoCompositor:
    """Composes final video from components."""
    
    def __init__(self, spec: VideoSpec, use_gpu: bool = False):
        self.spec = spec
        self.use_gpu = use_gpu
        self.temp_dir = Path("temp/video")
        self.temp_dir.mkdir(parents=True, exist_ok=True)
    
    def create_background(self, video_paths: List[str], duration: float) -> VideoFileClip:
        """Create background from video clips."""
        if not video_paths:
            # Create solid color background
            return ColorClip(
                size=(self.spec.width, self.spec.height),
                color=(20, 20, 30),
                duration=duration
            )
        
        clips = []
        total_duration = 0
        
        for video_path in video_paths:
            try:
                clip = VideoFileClip(video_path)
                
                # Resize to fit
                clip = self._resize_clip(clip)
                
                clips.append(clip)
                total_duration += clip.duration
                
                if total_duration >= duration:
                    break
            except Exception as e:
                logger.warning(f"Failed to load video {video_path}: {e}")
        
        if not clips:
            return ColorClip(
                size=(self.spec.width, self.spec.height),
                color=(20, 20, 30),
                duration=duration
            )
        
        # Concatenate and loop if needed
        if len(clips) == 1:
            background = clips[0]
        else:
            background = concatenate_videoclips(clips)
        
        # Loop to fill duration
        if background.duration < duration:
            loops_needed = math.ceil(duration / background.duration)
            background = background.loop(n=loops_needed)
        
        # Trim to exact duration
        background = background.subclip(0, duration)
        
        # Apply effects
        background = background.fx(vfx.colorx, 0.7)  # Darken slightly
        
        return background
    
    def create_background_from_images(self, image_paths: List[str], 
                                       duration: float) -> VideoFileClip:
        """Create background from images with Ken Burns effect."""
        if not image_paths:
            return ColorClip(
                size=(self.spec.width, self.spec.height),
                color=(20, 20, 30),
                duration=duration
            )
        
        clips = []
        time_per_image = duration / len(image_paths)
        
        for image_path in image_paths:
            try:
                clip = ImageClip(image_path, duration=time_per_image)
                clip = self._resize_clip(clip)
                
                # Add Ken Burns effect (slow zoom)
                clip = clip.resize(lambda t: 1 + 0.05 * t / time_per_image)
                
                clips.append(clip)
            except Exception as e:
                logger.warning(f"Failed to load image {image_path}: {e}")
        
        if not clips:
            return ColorClip(
                size=(self.spec.width, self.spec.height),
                color=(20, 20, 30),
                duration=duration
            )
        
        return concatenate_videoclips(clips, method="compose")
    
    def _resize_clip(self, clip) -> VideoFileClip:
        """Resize clip to fit video spec."""
        target_ratio = self.spec.width / self.spec.height
        clip_ratio = clip.w / clip.h
        
        if clip_ratio > target_ratio:
            # Clip is wider, fit by height
            new_height = self.spec.height
            new_width = int(clip.w * (new_height / clip.h))
        else:
            # Clip is taller, fit by width
            new_width = self.spec.width
            new_height = int(clip.h * (new_width / clip.w))
        
        clip = clip.resize((new_width, new_height))
        
        # Center crop
        x_center = new_width // 2
        y_center = new_height // 2
        
        clip = clip.crop(
            x_center=x_center,
            y_center=y_center,
            width=self.spec.width,
            height=self.spec.height
        )
        
        return clip
    
    def compose_video(self, background: VideoFileClip, 
                      audio: AudioFileClip,
                      subtitle_clips: List[TextClip],
                      output_path: str) -> str:
        """Compose final video with all elements."""
        
        # Set audio
        final_duration = audio.duration
        background = background.with_duration(final_duration)
        
        # Combine all clips
        all_clips = [background] + subtitle_clips
        
        # Create composite
        video = CompositeVideoClip(all_clips, size=(self.spec.width, self.spec.height))
        video = video.with_audio(audio)
        video = video.with_duration(final_duration)
        
        # Render
        logger.info(f"Rendering video to {output_path}")
        
        # FFmpeg settings for quality
        ffmpeg_params = [
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '23',
            '-c:a', 'aac',
            '-b:a', '192k'
        ]
        
        if self.use_gpu:
            # Try NVIDIA GPU encoding
            ffmpeg_params = [
                '-c:v', 'h264_nvenc',
                '-preset', 'medium',
                '-b:v', '5M',
                '-c:a', 'aac',
                '-b:a', '192k'
            ]
        
        video.write_videofile(
            output_path,
            fps=self.spec.fps,
            codec='libx264',
            audio_codec='aac',
            threads=4,
            preset='medium',
            ffmpeg_params=ffmpeg_params if not self.use_gpu else None
        )
        
        # Cleanup
        video.close()
        background.close()
        audio.close()
        
        return output_path


class VideoGenerator:
    """Main video generator that orchestrates the entire process."""
    
    def __init__(self, config):
        self.config = config
        
        # Parse resolution
        res = config.video.resolution.split('x')
        self.spec = VideoSpec(
            width=int(res[0]),
            height=int(res[1]),
            fps=config.video.fps,
            format=config.channel.video_format
        )
        
        # Initialize components
        self.subtitle_style = SubtitleStyle(
            font=config.subtitles.font,
            font_size=config.subtitles.font_size,
            color=config.subtitles.color,
            stroke_color=config.subtitles.stroke_color,
            stroke_width=config.subtitles.stroke_width,
            position=config.subtitles.position,
            word_highlight=config.subtitles.word_highlight,
            highlight_color=config.subtitles.highlight_color
        )
        
        self.subtitle_gen = SubtitleGenerator(self.subtitle_style)
        self.compositor = VideoCompositor(self.spec, config.video.use_gpu)
        
        self.output_dir = Path(config.storage.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate(self, script_text: str, audio_path: str,
                 background_videos: List[str] = None,
                 background_images: List[str] = None,
                 output_filename: str = None,
                 word_timestamps: List[Tuple[str, float, float]] = None) -> str:
        """
        Generate a complete video.
        
        Args:
            script_text: The script text for subtitles
            audio_path: Path to the audio file
            background_videos: List of background video paths
            background_images: List of background image paths (fallback)
            output_filename: Output filename (optional)
            word_timestamps: Word-level timestamps for precise subtitles
        
        Returns:
            Path to the generated video
        """
        logger.info("Starting video generation")
        
        # Load audio
        audio = AudioFileClip(audio_path)
        duration = audio.duration
        
        # Create background
        if background_videos:
            background = self.compositor.create_background(background_videos, duration)
        elif background_images:
            background = self.compositor.create_background_from_images(background_images, duration)
        else:
            background = ColorClip(
                size=(self.spec.width, self.spec.height),
                color=(20, 20, 30),
                duration=duration
            )
        
        # Create subtitles
        if word_timestamps:
            subtitle_clips = self.subtitle_gen.create_word_clips(
                word_timestamps, 
                (self.spec.width, self.spec.height)
            )
        else:
            subtitle_clips = self.subtitle_gen.create_sentence_clips(
                script_text,
                duration,
                (self.spec.width, self.spec.height)
            )
        
        # Generate output path
        if output_filename is None:
            import time
            output_filename = f"video_{int(time.time())}.mp4"
        
        output_path = str(self.output_dir / output_filename)
        
        # Compose final video
        result = self.compositor.compose_video(
            background,
            audio,
            subtitle_clips,
            output_path
        )
        
        logger.info(f"Video generated: {result}")
        
        return result
    
    def generate_from_components(self, 
                                  script,  # VideoScript object
                                  audio_result,  # AudioResult object
                                  media_manager) -> str:
        """
        Generate video from script, audio, and media components.
        High-level method that handles everything.
        """
        # Get background media
        if self.config.video.background_type == "stock_video":
            backgrounds = media_manager.get_background_videos(
                script.niche, 
                count=3,
                duration_needed=audio_result.duration
            )
            background_videos = backgrounds
            background_images = None
        else:
            backgrounds = media_manager.get_background_images(script.niche, count=5)
            background_videos = None
            background_images = backgrounds
        
        # Generate video
        output_filename = self._sanitize_filename(script.title) + ".mp4"
        
        return self.generate(
            script_text=script.full_script,
            audio_path=audio_result.audio_path,
            background_videos=background_videos,
            background_images=background_images,
            output_filename=output_filename,
            word_timestamps=audio_result.word_timestamps
        )
    
    def _sanitize_filename(self, title: str) -> str:
        """Sanitize title for use as filename."""
        # Remove special characters
        clean = re.sub(r'[^\w\s-]', '', title)
        # Replace spaces with underscores
        clean = re.sub(r'\s+', '_', clean)
        # Limit length
        return clean[:50]


def create_video_generator(config) -> VideoGenerator:
    """Factory function to create video generator."""
    return VideoGenerator(config)
