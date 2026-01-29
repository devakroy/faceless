"""
AI Background Video Generator
Generates short AI video clips from the script prompt.
Uses Diffusers Stable Video Diffusion (image-to-video) with an optional
text-to-image step to create the initial frame.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import logging
import re

from src.ai.script_generator import VideoScript

logger = logging.getLogger(__name__)


@dataclass
class AIVideoConfig:
    provider: str = "diffusers"
    video_model: str = "stabilityai/stable-video-diffusion-img2vid-xt"
    image_model: str = "stabilityai/sdxl-turbo"
    width: int = 384
    height: int = 640
    fps: int = 6
    clip_duration: float = 3.0
    max_frames: int = 12
    num_inference_steps: int = 6
    guidance_scale: float = 1.0
    seed: int = 42
    base_image_path: str = ""


class AIVideoBackgroundGenerator:
    """Generate AI background clips for videos."""

    def __init__(self, config):
        self.config = config
        self.ai_config = getattr(config, "ai_video", AIVideoConfig())
        self.output_dir = Path(config.storage.temp_dir) / "ai_backgrounds"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._image_pipe = None
        self._video_pipe = None
        self._device = None
        self._dtype = None

    def generate(self, script: VideoScript, duration: float) -> str:
        """Generate a short AI video clip and return its path."""
        if self.ai_config.provider != "diffusers":
            raise ValueError(f"Unknown AI video provider: {self.ai_config.provider}")

        # Aggressive memory cleanup before initialization
        self._cleanup_gpu_memory()
        
        self._init_pipelines()

        prompt = self._build_prompt(script)
        
        # Generate minimal initial image on GPU (required by Stable Video Diffusion)
        # Use very small resolution to save memory
        init_image = self._generate_minimal_init_image(prompt)

        fps = self.ai_config.fps or self.config.video.fps
        clip_duration = min(duration, self.ai_config.clip_duration)
        num_frames = min(self.ai_config.max_frames, max(6, int(fps * clip_duration)))

        logger.info(
            "Generating AI background video: %s frames @ %s fps",
            num_frames,
            fps,
        )

        generator = self._get_generator()
        result = self._video_pipe(
            image=init_image,
            num_frames=num_frames,
            decode_chunk_size=4,
            generator=generator,
        )
        frames = result.frames[0]

        from diffusers.utils import export_to_video

        output_path = self.output_dir / f"ai_bg_{abs(hash(prompt)) % 100000}.mp4"
        export_to_video(frames, str(output_path), fps=fps)

        return str(output_path)

    def _build_prompt(self, script: VideoScript) -> str:
        """Build a visual prompt from the script."""
        keywords = self._extract_keywords(script.full_script, script.keywords)
        keywords_str = ", ".join(keywords[:8]) if keywords else script.niche
        return (
            f"{script.title}. {script.hook} {script.body} "
            f"Key visuals: {keywords_str}. "
            f"Cinematic b-roll, clear subject, {script.niche} theme, vertical video, "
            f"high quality, realistic, coherent scene."
        )

    def _extract_keywords(self, text: str, fallback: Optional[list] = None) -> list:
        """Extract compact keywords from script text for better visual relevance."""
        words = re.findall(r"[A-Za-z0-9']+", text.lower())
        stopwords = {
            "the", "and", "that", "with", "this", "from", "your", "you",
            "are", "was", "were", "have", "has", "had", "will", "just",
            "like", "what", "when", "then", "than", "them", "they", "their",
            "our", "for", "but", "not", "all", "can", "could", "should",
            "about", "into", "over", "under", "more", "most", "very", "really",
        }
        keywords = [w for w in words if len(w) > 3 and w not in stopwords]
        if keywords:
            # keep order but unique
            seen = set()
            unique = []
            for w in keywords:
                if w not in seen:
                    unique.append(w)
                    seen.add(w)
            return unique[:12]
        return fallback or []

    def _get_init_image(self, prompt: str):
        """Create or load the initial image for image-to-video."""
        # This method is now deprecated - we generate video directly from text
        # Keeping for potential future use
        if self.ai_config.base_image_path:
            from PIL import Image
            return Image.open(self.ai_config.base_image_path).convert("RGB")
        
        # Fallback: generate a simple placeholder if needed
        logger.warning("Generating fallback image - this shouldn't be needed")
        from PIL import Image
        return Image.new('RGB', (self.ai_config.width, self.ai_config.height), color='black')

    def _init_pipelines(self):
        """Lazy init diffusers pipelines."""
        if self._video_pipe:
            return

        try:
            import torch
            from diffusers import AutoPipelineForText2Image, StableVideoDiffusionPipeline
        except Exception as e:
            raise RuntimeError(
                "AI video generation requires diffusers + torch.\n"
                "Install: pip install diffusers torch accelerate safetensors"
            ) from e

        if torch.cuda.is_available():
            self._device = "cuda"
            self._dtype = torch.float16
            logger.info("Using CUDA GPU for AI video generation")
        else:
            raise RuntimeError(
                "CUDA GPU is required for AI video generation. "
                "Please ensure you have an NVIDIA GPU with CUDA drivers installed."
            )

        # Load both pipelines but optimize for memory
        logger.info("Loading AI image model: %s", self.ai_config.image_model)
        self._image_pipe = AutoPipelineForText2Image.from_pretrained(
            self.ai_config.image_model,
            torch_dtype=self._dtype,
        ).to(self._device)
        if hasattr(self._image_pipe, "enable_attention_slicing"):
            self._image_pipe.enable_attention_slicing()

        logger.info("Loading AI video model: %s", self.ai_config.video_model)
        self._video_pipe = StableVideoDiffusionPipeline.from_pretrained(
            self.ai_config.video_model,
            torch_dtype=self._dtype,
        ).to(self._device)
        
        # Enable memory efficient attention
        if hasattr(self._video_pipe, 'enable_model_cpu_offload'):
            self._video_pipe.enable_model_cpu_offload()
        
        # Enable xformers for memory efficiency
        try:
            self._video_pipe.enable_xformers_memory_efficient_attention()
        except Exception as e:
            logger.info(f"XFormers not available: {e}")
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Clear GPU cache
        import torch
        torch.cuda.empty_cache()
        
    def _cleanup_gpu_memory(self):
        """Aggressively clean up GPU memory."""
        try:
            import torch
            import gc
            
            # Clear GPU cache
            torch.cuda.empty_cache()
            
            # Force garbage collection
            gc.collect()
            
            # Reset CUDA device
            if hasattr(torch.cuda, 'reset_peak_memory_stats'):
                torch.cuda.reset_peak_memory_stats()
            
            logger.info("GPU memory cleanup completed")
            
        except Exception as e:
            logger.warning(f"GPU memory cleanup failed: {e}")
    
    def _generate_minimal_init_image(self, prompt: str):
        """Generate a minimal initial image to use as input for video generation."""
        logger.info("Generating minimal init image from prompt")
        
        # Use very small resolution to save memory
        small_width = 256  # Much smaller than original
        small_height = 256
        
        # Clear GPU cache before generation
        import torch
        torch.cuda.empty_cache()
        
        image = self._image_pipe(
            prompt=prompt,
            height=small_height,
            width=small_width,
            num_inference_steps=4,  # Fewer steps = less memory
            guidance_scale=1.0,
            generator=self._get_generator(),
        ).images[0]
        
        # Resize to target resolution for video generation
        target_width = self.ai_config.width
        target_height = self.ai_config.height
        image = image.resize((target_width, target_height))
        
        # Clear GPU cache after generation
        torch.cuda.empty_cache()
        
        return image
        if hasattr(self._video_pipe, "enable_attention_slicing"):
            self._video_pipe.enable_attention_slicing()
        if hasattr(self._video_pipe, "enable_vae_slicing"):
            self._video_pipe.enable_vae_slicing()
        if self._device == "cuda" and hasattr(self._video_pipe, "enable_model_cpu_offload"):
            self._video_pipe.enable_model_cpu_offload()

    def _get_generator(self):
        """Get torch generator for deterministic outputs."""
        import torch

        return torch.Generator(device=self._device).manual_seed(self.ai_config.seed)


def create_ai_background_generator(config) -> AIVideoBackgroundGenerator:
    """Factory for AI background generator."""
    return AIVideoBackgroundGenerator(config)
