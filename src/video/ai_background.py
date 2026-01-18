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

        self._init_pipelines()

        prompt = self._build_prompt(script)
        init_image = self._get_init_image(prompt)

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
        return (
            f"{script.title}. {script.hook} "
            f"Cinematic b-roll, {script.niche} theme, high quality, vertical video."
        )

    def _get_init_image(self, prompt: str):
        """Create or load the initial image for image-to-video."""
        if self.ai_config.base_image_path:
            from PIL import Image

            return Image.open(self.ai_config.base_image_path).convert("RGB")

        logger.info("Generating AI init image from prompt")
        image = self._image_pipe(
            prompt=prompt,
            height=self.ai_config.height,
            width=self.ai_config.width,
            num_inference_steps=self.ai_config.num_inference_steps,
            guidance_scale=self.ai_config.guidance_scale,
            generator=self._get_generator(),
        ).images[0]
        return image

    def _init_pipelines(self):
        """Lazy init diffusers pipelines."""
        if self._image_pipe and self._video_pipe:
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
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            self._device = "mps"
            self._dtype = torch.float16
        else:
            self._device = "cpu"
            self._dtype = torch.float32
            logger.warning("AI video generation on CPU will be very slow.")

        logger.info("Loading AI image model: %s", self.ai_config.image_model)
        self._image_pipe = AutoPipelineForText2Image.from_pretrained(
            self.ai_config.image_model,
            dtype=self._dtype,
        ).to(self._device)
        if hasattr(self._image_pipe, "enable_attention_slicing"):
            self._image_pipe.enable_attention_slicing()

        logger.info("Loading AI video model: %s", self.ai_config.video_model)
        self._video_pipe = StableVideoDiffusionPipeline.from_pretrained(
            self.ai_config.video_model,
            dtype=self._dtype,
        ).to(self._device)
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
