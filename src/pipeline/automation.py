"""
Main Automation Pipeline
Orchestrates the entire video creation and upload process.
Runs fully automated on a schedule.
"""

import os
import time
import schedule
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import json
import threading

from src.utils.config_loader import get_config, Config
from src.ai.script_generator import create_script_generator, VideoScript
from src.audio.tts_engine import create_tts_engine, AudioResult
from src.media.stock_media import create_stock_media_manager
from src.video.video_generator import create_video_generator
from src.video.ai_background import create_ai_background_generator
from src.youtube.uploader import create_youtube_uploader, VideoMetadata
from src.seo.optimizer import create_seo_optimizer
from src.analytics.tracker import create_analytics_tracker

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Result of a pipeline run."""
    success: bool
    video_id: Optional[str] = None
    video_path: Optional[str] = None
    title: Optional[str] = None
    error: Optional[str] = None
    duration: float = 0.0
    

class ContentPipeline:
    """
    Main content creation pipeline.
    Handles the entire flow from script generation to video upload.
    """
    
    def __init__(self, config: Config):
        self.config = config
        
        # Initialize all components
        logger.info("Initializing pipeline components...")
        
        self.script_generator = create_script_generator(config)
        self.tts_engine = create_tts_engine(config)
        self.media_manager = create_stock_media_manager(config)
        self.video_generator = create_video_generator(config)
        self.seo_optimizer = create_seo_optimizer(config)
        self.youtube_uploader = create_youtube_uploader(config)
        self.analytics = create_analytics_tracker(config)
        self.ai_background_generator = None
        
        # Pipeline state
        self.is_running = False
        self.videos_created_today = 0
        self.last_run_time = None
        
        logger.info("Pipeline initialized successfully")
    
    def run_once(self, niche: Optional[str] = None) -> PipelineResult:
        """
        Run the pipeline once to create and upload a single video.
        
        Args:
            niche: Optional niche override (uses config default if not specified)
        
        Returns:
            PipelineResult with details of the operation
        """
        start_time = time.time()
        niche = niche or self.config.channel.niche
        
        logger.info(f"Starting pipeline run for niche: {niche}")
        
        try:
            # Step 1: Generate script
            logger.info("Step 1: Generating script...")
            script = self.script_generator.generate_script(
                niche=niche,
                duration=self.config.channel.target_duration
            )
            logger.info(f"Script generated: {script.title}")
            
            # Step 2: Generate audio
            logger.info("Step 2: Generating audio...")
            audio_result = self.tts_engine.generate_with_timestamps(script.full_script)
            logger.info(f"Audio generated: {audio_result.duration:.2f}s")
            
            # Step 3: Get background media
            logger.info("Step 3: Fetching background media...")
            if self.config.video.background_type == "ai_generated":
                try:
                    if self.ai_background_generator is None:
                        self.ai_background_generator = create_ai_background_generator(self.config)
                    ai_video_path = self.ai_background_generator.generate(script, audio_result.duration)
                    backgrounds = [ai_video_path]
                except Exception as e:
                    logger.warning(f"AI background generation failed, falling back to stock: {e}")
                    backgrounds = self.media_manager.get_background_videos(
                        niche, count=3, duration_needed=audio_result.duration
                    )
            elif self.config.video.background_type == "stock_video":
                backgrounds = self.media_manager.get_background_videos(
                    niche, count=3, duration_needed=audio_result.duration
                )
            else:
                backgrounds = self.media_manager.get_background_images(niche, count=5)
            logger.info(f"Got {len(backgrounds)} background media files")

            # Split backgrounds into videos/images based on extension
            video_exts = {".mp4", ".mov", ".mkv", ".webm", ".avi"}
            image_exts = {".jpg", ".jpeg", ".png", ".webp"}
            background_videos = []
            background_images = []
            for path in backgrounds:
                ext = Path(path).suffix.lower()
                if ext in video_exts:
                    background_videos.append(path)
                elif ext in image_exts:
                    background_images.append(path)
            
            # Step 4: Generate video
            logger.info("Step 4: Generating video...")
            video_path = self.video_generator.generate(
                script_text=script.full_script,
                audio_path=audio_result.audio_path,
                background_videos=background_videos or None,
                background_images=background_images or None,
                word_timestamps=audio_result.word_timestamps
            )
            logger.info(f"Video generated: {video_path}")
            
            # Step 5: SEO optimization
            logger.info("Step 5: Optimizing SEO...")
            seo_result = self.seo_optimizer.optimize(
                title=script.title,
                hook=script.hook,
                keywords=script.keywords,
                niche=niche,
                background_image=background_images[0] if background_images else None
            )
            logger.info(f"SEO optimized: {seo_result.title}")
            
            # Step 6: Upload to YouTube
            logger.info("Step 6: Uploading to YouTube...")
            metadata = VideoMetadata(
                title=seo_result.title,
                description=seo_result.description,
                tags=seo_result.tags,
                category_id=self.config.youtube.category_id,
                privacy_status=self.config.youtube.privacy_status,
                thumbnail_path=seo_result.thumbnail_path
            )
            
            upload_result = self.youtube_uploader.upload_video(video_path, metadata)
            
            if not upload_result.success:
                raise Exception(f"Upload failed: {upload_result.error}")
            
            logger.info(f"Video uploaded: {upload_result.video_url}")
            
            # Step 7: Track analytics
            logger.info("Step 7: Tracking analytics...")
            self.analytics.track_video(
                video_id=upload_result.video_id,
                title=seo_result.title,
                niche=niche,
                duration=int(audio_result.duration),
                description=seo_result.description,
                thumbnail_path=seo_result.thumbnail_path
            )
            
            # Update counters
            self.videos_created_today += 1
            self.last_run_time = datetime.now()
            
            duration = time.time() - start_time
            logger.info(f"Pipeline completed in {duration:.2f}s")
            
            return PipelineResult(
                success=True,
                video_id=upload_result.video_id,
                video_path=video_path,
                title=seo_result.title,
                duration=duration
            )
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            return PipelineResult(
                success=False,
                error=str(e),
                duration=time.time() - start_time
            )
    
    def run_batch(self, count: int = 3, niches: Optional[List[str]] = None) -> List[PipelineResult]:
        """
        Run the pipeline multiple times to create a batch of videos.
        
        Args:
            count: Number of videos to create
            niches: Optional list of niches to cycle through
        
        Returns:
            List of PipelineResults
        """
        results = []
        niches = niches or [self.config.channel.niche]
        
        for i in range(count):
            niche = niches[i % len(niches)]
            logger.info(f"Creating video {i+1}/{count} for niche: {niche}")
            
            result = self.run_once(niche)
            results.append(result)
            
            if result.success:
                logger.info(f"Video {i+1} created successfully")
            else:
                logger.error(f"Video {i+1} failed: {result.error}")
            
            # Small delay between videos
            if i < count - 1:
                time.sleep(5)
        
        return results


class AutomationScheduler:
    """
    Schedules and manages automated content creation.
    Runs continuously in the background.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.pipeline = ContentPipeline(config)
        self.is_running = False
        self._thread: Optional[threading.Thread] = None
        
        # Load state
        self.state_file = Path(config.storage.output_dir) / "scheduler_state.json"
        self.state = self._load_state()
    
    def _load_state(self) -> Dict[str, Any]:
        """Load scheduler state from file."""
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                return json.load(f)
        return {
            'videos_today': 0,
            'last_video_date': None,
            'total_videos': 0,
            'last_run': None
        }
    
    def _save_state(self):
        """Save scheduler state to file."""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2, default=str)
    
    def _reset_daily_counter(self):
        """Reset daily video counter if it's a new day."""
        today = datetime.now().date().isoformat()
        if self.state.get('last_video_date') != today:
            self.state['videos_today'] = 0
            self.state['last_video_date'] = today
            self._save_state()
    
    def _can_create_video(self) -> bool:
        """Check if we can create another video today."""
        self._reset_daily_counter()
        max_per_day = self.config.youtube.upload_times.__len__() if hasattr(self.config.youtube, 'upload_times') else 3
        return self.state['videos_today'] < max_per_day
    
    def _scheduled_run(self):
        """Run scheduled video creation."""
        if not self._can_create_video():
            logger.info("Daily video limit reached, skipping")
            return
        
        logger.info("Starting scheduled video creation")
        result = self.pipeline.run_once()
        
        if result.success:
            self.state['videos_today'] += 1
            self.state['total_videos'] += 1
            self.state['last_run'] = datetime.now().isoformat()
            self._save_state()
            logger.info(f"Scheduled video created: {result.title}")
        else:
            logger.error(f"Scheduled video failed: {result.error}")
    
    def setup_schedule(self):
        """Set up the upload schedule."""
        upload_times = getattr(self.config.youtube, 'upload_times', ["09:00", "15:00", "21:00"])
        
        for upload_time in upload_times:
            schedule.every().day.at(upload_time).do(self._scheduled_run)
            logger.info(f"Scheduled video creation at {upload_time}")
    
    def _run_scheduler(self):
        """Run the scheduler loop."""
        self.setup_schedule()
        
        while self.is_running:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    def start(self):
        """Start the automation scheduler."""
        if self.is_running:
            logger.warning("Scheduler is already running")
            return
        
        self.is_running = True
        self._thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self._thread.start()
        logger.info("Automation scheduler started")
    
    def stop(self):
        """Stop the automation scheduler."""
        self.is_running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Automation scheduler stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current scheduler status."""
        self._reset_daily_counter()
        
        return {
            'is_running': self.is_running,
            'videos_today': self.state['videos_today'],
            'total_videos': self.state['total_videos'],
            'last_run': self.state.get('last_run'),
            'next_scheduled': self._get_next_scheduled_time()
        }
    
    def _get_next_scheduled_time(self) -> Optional[str]:
        """Get the next scheduled run time."""
        jobs = schedule.get_jobs()
        if jobs:
            next_run = min(job.next_run for job in jobs)
            return next_run.isoformat()
        return None


class FacelessAutomation:
    """
    Main automation class that provides a simple interface
    for running the faceless YouTube channel.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the automation system."""
        from src.utils.config_loader import ConfigLoader
        
        self.config_loader = ConfigLoader(config_path)
        self.config = self.config_loader.load()
        self.config_loader.ensure_directories()
        
        self.scheduler = AutomationScheduler(self.config)
        self.pipeline = self.scheduler.pipeline
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / "automation.log"),
                logging.StreamHandler()
            ]
        )
    
    def create_video(self, niche: Optional[str] = None) -> PipelineResult:
        """Create a single video."""
        return self.pipeline.run_once(niche)
    
    def create_batch(self, count: int = 3, niches: Optional[List[str]] = None) -> List[PipelineResult]:
        """Create multiple videos."""
        return self.pipeline.run_batch(count, niches)
    
    def start_automation(self):
        """Start fully automated mode."""
        self.scheduler.start()
        logger.info("Faceless automation started - running 24/7")
    
    def stop_automation(self):
        """Stop automated mode."""
        self.scheduler.stop()
        logger.info("Faceless automation stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current automation status."""
        return {
            'scheduler': self.scheduler.get_status(),
            'channel': {
                'name': self.config.channel.name,
                'niche': self.config.channel.niche,
                'format': self.config.channel.video_format
            },
            'analytics': self.pipeline.analytics.get_dashboard_data()
        }
    
    def get_insights(self) -> List[Dict]:
        """Get growth insights."""
        insights = self.pipeline.analytics.get_insights()
        return [
            {
                'category': i.category,
                'insight': i.insight,
                'action': i.action,
                'priority': i.priority
            }
            for i in insights
        ]
    
    def update_channel_stats(self):
        """Fetch and update channel statistics from YouTube."""
        channel_info = self.pipeline.youtube_uploader.get_channel_info()
        if channel_info:
            self.pipeline.analytics.update_channel_stats(
                subscribers=int(channel_info.get('subscribers', 0)),
                total_views=int(channel_info.get('views', 0)),
                total_videos=int(channel_info.get('videos', 0))
            )
            logger.info(f"Updated channel stats: {channel_info['subscribers']} subscribers")


def create_automation(config_path: str = "config/config.yaml") -> FacelessAutomation:
    """Factory function to create the automation system."""
    return FacelessAutomation(config_path)
