"""
AI Image Background Generator
Generates AI images per script sentence for better script-to-visual sync.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List
import logging

from src.ai.script_generator import VideoScript

logger = logging.getLogger(__name__)


@dataclass
class AIImageConfig:
    provider: str = "diffusers"
    image_model: str = "stabilityai/sdxl-turbo"
    width: int = 512
    height: int = 896
    num_inference_steps: int = 6
    guidance_scale: float = 1.0
    seed: int = 42


class AIImageBackgroundGenerator:
    """Generate AI images to match script sentences."""

    def __init__(self, config):
        self.config = config
        self.ai_config = getattr(config, "ai_image", AIImageConfig())
        self.output_dir = Path(config.storage.temp_dir) / "ai_images"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._image_pipe = None
        self._device = None
        self._dtype = None

    def generate_for_sentences(self, script: VideoScript, sentences: List[str]) -> List[str]:
        if self.ai_config.provider != "diffusers":
            raise ValueError(f"Unknown AI image provider: {self.ai_config.provider}")

        # Aggressive memory cleanup before initialization
        self._cleanup_gpu_memory()
        
        self._init_pipeline()

        image_paths: List[str] = []
        for idx, sentence in enumerate(sentences):
            prompt = self._build_prompt(script, sentence)
            # Clear GPU cache before generation
            import torch
            torch.cuda.empty_cache()
            
            image = self._image_pipe(
                prompt=prompt,
                height=self.ai_config.height,
                width=self.ai_config.width,
                num_inference_steps=self.ai_config.num_inference_steps,
                guidance_scale=self.ai_config.guidance_scale,
                generator=self._get_generator(),
            ).images[0]
            
            # Clear GPU cache after generation
            torch.cuda.empty_cache()
            output_path = self.output_dir / f"ai_img_{idx}_{abs(hash(prompt)) % 100000}.png"
            image.save(output_path)
            image_paths.append(str(output_path))

        return image_paths

    def _build_prompt(self, script: VideoScript, sentence: str) -> str:
        return (
            f"{sentence}. Visual style: cinematic, high quality, vertical frame, "
            f"{script.niche} theme."
        )

    def _init_pipeline(self):
        if self._image_pipe is not None:
            return

        try:
            import torch
            from diffusers import AutoPipelineForText2Image
        except Exception as e:
            raise RuntimeError(
                "AI image generation requires diffusers + torch.\n"
                "Install: pip install diffusers torch accelerate safetensors"
            ) from e

        if torch.cuda.is_available():
            self._device = "cuda"
            self._dtype = torch.float16
            logger.info("Using CUDA GPU for AI image generation")
        else:
            raise RuntimeError(
                "CUDA GPU is required for AI image generation. "
                "Please ensure you have an NVIDIA GPU with CUDA drivers installed."
            )

        logger.info("Loading AI image model: %s", self.ai_config.image_model)
        self._image_pipe = AutoPipelineForText2Image.from_pretrained(
            self.ai_config.image_model,
            torch_dtype=self._dtype,
        ).to(self._device)
        
        # Enable memory efficient attention
        if hasattr(self._image_pipe, 'enable_model_cpu_offload'):
            self._image_pipe.enable_model_cpu_offload()
        
        # Enable xformers for memory efficiency
        try:
            self._image_pipe.enable_xformers_memory_efficient_attention()
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
        if hasattr(self._image_pipe, "enable_attention_slicing"):
            self._image_pipe.enable_attention_slicing()

    def _get_generator(self):
        import torch

        return torch.Generator(device=self._device).manual_seed(self.ai_config.seed)


def create_ai_image_background_generator(config) -> AIImageBackgroundGenerator:
    return AIImageBackgroundGenerator(config)
"""
AI Image Background Generator
Generates AI images per script sentence for better script-to-visual sync.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List
import logging

from src.ai.script_generator import VideoScript

logger = logging.getLogger(__name__)


@dataclass
class AIImageConfig:
    provider: str = "diffusers"
    image_model: str = "stabilityai/sdxl-turbo"
    width: int = 512
    height: int = 896
    num_inference_steps: int = 6
    guidance_scale: float = 1.0
    seed: int = 42


class AIImageBackgroundGenerator:
    """Generate AI images to match script sentences."""

    def __init__(self, config):
        self.config = config
        self.ai_config = getattr(config, "ai_image", AIImageConfig())
        self.output_dir = Path(config.storage.temp_dir) / "ai_images"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._image_pipe = None
        self._device = None
        self._dtype = None

    def generate_for_sentences(self, script: VideoScript, sentences: List[str]) -> List[str]:
        if self.ai_config.provider != "diffusers":
            raise ValueError(f"Unknown AI image provider: {self.ai_config.provider}")

        self._init_pipeline()

        image_paths: List[str] = []
        for idx, sentence in enumerate(sentences):
            prompt = self._build_prompt(script, sentence)
            image = self._image_pipe(
                prompt=prompt,
                height=self.ai_config.height,
                width=self.ai_config.width,
                num_inference_steps=self.ai_config.num_inference_steps,
                guidance_scale=self.ai_config.guidance_scale,
                generator=self._get_generator(),
            ).images[0]
            output_path = self.output_dir / f"ai_img_{idx}_{abs(hash(prompt)) % 100000}.png"
            image.save(output_path)
            image_paths.append(str(output_path))

        return image_paths

    def _build_prompt(self, script: VideoScript, sentence: str) -> str:
        return (
            f"{sentence}. Visual style: cinematic, high quality, vertical frame, "
            f"{script.niche} theme."
        )

    def _init_pipeline(self):
        if self._image_pipe is not None:
            return

        try:
            import torch
            from diffusers import AutoPipelineForText2Image
        except Exception as e:
            raise RuntimeError(
                "AI image generation requires diffusers + torch.\n"
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
            logger.warning("AI image generation on CPU will be slow.")

        logger.info("Loading AI image model: %s", self.ai_config.image_model)
        self._image_pipe = AutoPipelineForText2Image.from_pretrained(
            self.ai_config.image_model,
            dtype=self._dtype,
        ).to(self._device)
        if hasattr(self._image_pipe, "enable_attention_slicing"):
            self._image_pipe.enable_attention_slicing()

    def _get_generator(self):
        import torch

        return torch.Generator(device=self._device).manual_seed(self.ai_config.seed)


def create_ai_image_background_generator(config) -> AIImageBackgroundGenerator:
    return AIImageBackgroundGenerator(config)
