"""
YouTube Upload Automation
Uses YouTube Data API v3 (FREE - 10,000 quota units/day)
Handles OAuth2 authentication, video upload, and metadata management.
"""

import os
import json
import pickle
import time
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from googleapiclient.errors import HttpError

logger = logging.getLogger(__name__)


# YouTube API scopes
SCOPES = [
    'https://www.googleapis.com/auth/youtube.upload',
    'https://www.googleapis.com/auth/youtube',
    'https://www.googleapis.com/auth/youtube.readonly'
]


@dataclass
class VideoMetadata:
    """Metadata for YouTube video upload."""
    title: str
    description: str
    tags: List[str]
    category_id: str = "22"  # People & Blogs
    privacy_status: str = "public"  # public, private, unlisted
    made_for_kids: bool = False
    scheduled_time: Optional[datetime] = None
    thumbnail_path: Optional[str] = None
    playlist_id: Optional[str] = None


@dataclass
class UploadResult:
    """Result of a video upload."""
    success: bool
    video_id: Optional[str] = None
    video_url: Optional[str] = None
    error: Optional[str] = None
    quota_used: int = 0


class YouTubeAuthenticator:
    """Handles YouTube OAuth2 authentication."""
    
    def __init__(self, client_secrets_file: str, credentials_file: str):
        self.client_secrets_file = Path(client_secrets_file)
        self.credentials_file = Path(credentials_file)
        self.credentials: Optional[Credentials] = None
    
    def authenticate(self) -> Credentials:
        """Authenticate with YouTube API."""
        # Check for existing credentials
        if self.credentials_file.exists():
            with open(self.credentials_file, 'rb') as f:
                self.credentials = pickle.load(f)
        
        # Refresh or get new credentials
        if self.credentials and self.credentials.expired and self.credentials.refresh_token:
            logger.info("Refreshing expired credentials")
            self.credentials.refresh(Request())
        elif not self.credentials or not self.credentials.valid:
            if not self.client_secrets_file.exists():
                raise FileNotFoundError(
                    f"Client secrets file not found: {self.client_secrets_file}\n"
                    "Please download it from Google Cloud Console."
                )
            
            logger.info("Starting OAuth2 flow")
            flow = InstalledAppFlow.from_client_secrets_file(
                str(self.client_secrets_file),
                SCOPES
            )
            self.credentials = flow.run_local_server(port=8080)
        
        # Save credentials
        self.credentials_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.credentials_file, 'wb') as f:
            pickle.dump(self.credentials, f)
        
        logger.info("Authentication successful")
        return self.credentials
    
    def get_service(self):
        """Get authenticated YouTube service."""
        if not self.credentials:
            self.authenticate()
        
        return build('youtube', 'v3', credentials=self.credentials)


class YouTubeUploader:
    """Handles video uploads to YouTube."""
    
    # Retry settings
    MAX_RETRIES = 3
    RETRY_DELAY = 5
    
    # Quota costs (approximate)
    QUOTA_UPLOAD = 1600
    QUOTA_UPDATE = 50
    QUOTA_THUMBNAIL = 50
    
    def __init__(self, config):
        self.config = config
        self.authenticator = YouTubeAuthenticator(
            config.youtube.client_secrets_file,
            config.youtube.credentials_file
        )
        self._service = None
        self.daily_quota_used = 0
        self.daily_quota_limit = 10000
    
    @property
    def service(self):
        """Lazy load YouTube service."""
        if self._service is None:
            self._service = self.authenticator.get_service()
        return self._service
    
    def upload_video(self, video_path: str, metadata: VideoMetadata) -> UploadResult:
        """
        Upload a video to YouTube.
        
        Args:
            video_path: Path to the video file
            metadata: Video metadata
        
        Returns:
            UploadResult with video ID and URL
        """
        if not os.path.exists(video_path):
            return UploadResult(success=False, error=f"Video file not found: {video_path}")
        
        # Check quota
        if self.daily_quota_used + self.QUOTA_UPLOAD > self.daily_quota_limit:
            return UploadResult(
                success=False, 
                error="Daily quota limit reached. Try again tomorrow."
            )
        
        # Prepare video body
        body = {
            'snippet': {
                'title': metadata.title[:100],  # Max 100 chars
                'description': metadata.description[:5000],  # Max 5000 chars
                'tags': metadata.tags[:500],  # Max 500 tags
                'categoryId': metadata.category_id
            },
            'status': {
                'privacyStatus': metadata.privacy_status,
                'selfDeclaredMadeForKids': metadata.made_for_kids
            }
        }
        
        # Add scheduled publish time if specified
        if metadata.scheduled_time and metadata.privacy_status == "private":
            body['status']['publishAt'] = metadata.scheduled_time.isoformat() + 'Z'
            body['status']['privacyStatus'] = 'private'
        
        # Create media upload
        media = MediaFileUpload(
            video_path,
            mimetype='video/mp4',
            resumable=True,
            chunksize=1024 * 1024  # 1MB chunks
        )
        
        # Upload with retry
        for attempt in range(self.MAX_RETRIES):
            try:
                logger.info(f"Uploading video: {metadata.title} (attempt {attempt + 1})")
                
                request = self.service.videos().insert(
                    part='snippet,status',
                    body=body,
                    media_body=media
                )
                
                response = self._resumable_upload(request)
                
                if response:
                    video_id = response['id']
                    video_url = f"https://youtube.com/watch?v={video_id}"
                    
                    self.daily_quota_used += self.QUOTA_UPLOAD
                    
                    logger.info(f"Upload successful: {video_url}")
                    
                    # Upload thumbnail if provided
                    if metadata.thumbnail_path:
                        self._upload_thumbnail(video_id, metadata.thumbnail_path)
                    
                    # Add to playlist if specified
                    if metadata.playlist_id:
                        self._add_to_playlist(video_id, metadata.playlist_id)
                    
                    return UploadResult(
                        success=True,
                        video_id=video_id,
                        video_url=video_url,
                        quota_used=self.QUOTA_UPLOAD
                    )
                
            except HttpError as e:
                logger.error(f"HTTP error during upload: {e}")
                if e.resp.status in [500, 502, 503, 504]:
                    # Retry on server errors
                    time.sleep(self.RETRY_DELAY * (attempt + 1))
                    continue
                else:
                    return UploadResult(success=False, error=str(e))
            
            except Exception as e:
                logger.error(f"Upload error: {e}")
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(self.RETRY_DELAY)
                    continue
                return UploadResult(success=False, error=str(e))
        
        return UploadResult(success=False, error="Max retries exceeded")
    
    def _resumable_upload(self, request) -> Optional[Dict]:
        """Handle resumable upload with progress tracking."""
        response = None
        
        while response is None:
            try:
                status, response = request.next_chunk()
                
                if status:
                    progress = int(status.progress() * 100)
                    logger.info(f"Upload progress: {progress}%")
                    
            except HttpError as e:
                if e.resp.status in [500, 502, 503, 504]:
                    time.sleep(5)
                    continue
                raise
        
        return response
    
    def _upload_thumbnail(self, video_id: str, thumbnail_path: str) -> bool:
        """Upload custom thumbnail for a video."""
        if not os.path.exists(thumbnail_path):
            logger.warning(f"Thumbnail not found: {thumbnail_path}")
            return False
        
        try:
            self.service.thumbnails().set(
                videoId=video_id,
                media_body=MediaFileUpload(thumbnail_path, mimetype='image/jpeg')
            ).execute()
            
            self.daily_quota_used += self.QUOTA_THUMBNAIL
            logger.info(f"Thumbnail uploaded for video: {video_id}")
            return True
            
        except HttpError as e:
            logger.error(f"Thumbnail upload failed: {e}")
            return False
    
    def _add_to_playlist(self, video_id: str, playlist_id: str) -> bool:
        """Add video to a playlist."""
        try:
            self.service.playlistItems().insert(
                part='snippet',
                body={
                    'snippet': {
                        'playlistId': playlist_id,
                        'resourceId': {
                            'kind': 'youtube#video',
                            'videoId': video_id
                        }
                    }
                }
            ).execute()
            
            logger.info(f"Added video {video_id} to playlist {playlist_id}")
            return True
            
        except HttpError as e:
            logger.error(f"Failed to add to playlist: {e}")
            return False
    
    def update_video(self, video_id: str, metadata: VideoMetadata) -> bool:
        """Update video metadata."""
        try:
            body = {
                'id': video_id,
                'snippet': {
                    'title': metadata.title[:100],
                    'description': metadata.description[:5000],
                    'tags': metadata.tags[:500],
                    'categoryId': metadata.category_id
                }
            }
            
            self.service.videos().update(
                part='snippet',
                body=body
            ).execute()
            
            self.daily_quota_used += self.QUOTA_UPDATE
            logger.info(f"Updated video: {video_id}")
            return True
            
        except HttpError as e:
            logger.error(f"Update failed: {e}")
            return False
    
    def get_channel_info(self) -> Optional[Dict]:
        """Get information about the authenticated channel."""
        try:
            response = self.service.channels().list(
                part='snippet,statistics',
                mine=True
            ).execute()
            
            if response.get('items'):
                channel = response['items'][0]
                return {
                    'id': channel['id'],
                    'title': channel['snippet']['title'],
                    'subscribers': channel['statistics'].get('subscriberCount', 0),
                    'views': channel['statistics'].get('viewCount', 0),
                    'videos': channel['statistics'].get('videoCount', 0)
                }
            
            return None
            
        except HttpError as e:
            logger.error(f"Failed to get channel info: {e}")
            return None
    
    def get_video_analytics(self, video_id: str) -> Optional[Dict]:
        """Get basic analytics for a video."""
        try:
            response = self.service.videos().list(
                part='statistics,snippet',
                id=video_id
            ).execute()
            
            if response.get('items'):
                video = response['items'][0]
                stats = video['statistics']
                return {
                    'title': video['snippet']['title'],
                    'views': int(stats.get('viewCount', 0)),
                    'likes': int(stats.get('likeCount', 0)),
                    'comments': int(stats.get('commentCount', 0)),
                    'published_at': video['snippet']['publishedAt']
                }
            
            return None
            
        except HttpError as e:
            logger.error(f"Failed to get video analytics: {e}")
            return None
    
    def create_playlist(self, title: str, description: str = "") -> Optional[str]:
        """Create a new playlist."""
        try:
            response = self.service.playlists().insert(
                part='snippet,status',
                body={
                    'snippet': {
                        'title': title,
                        'description': description
                    },
                    'status': {
                        'privacyStatus': 'public'
                    }
                }
            ).execute()
            
            playlist_id = response['id']
            logger.info(f"Created playlist: {title} ({playlist_id})")
            return playlist_id
            
        except HttpError as e:
            logger.error(f"Failed to create playlist: {e}")
            return None


class UploadScheduler:
    """Schedules video uploads at optimal times."""
    
    # Best times to post (in UTC)
    OPTIMAL_HOURS = [9, 12, 15, 18, 21]  # 9am, 12pm, 3pm, 6pm, 9pm
    
    def __init__(self, uploader: YouTubeUploader, config):
        self.uploader = uploader
        self.config = config
        self.upload_queue: List[Dict] = []
    
    def queue_upload(self, video_path: str, metadata: VideoMetadata, 
                     scheduled_time: Optional[datetime] = None):
        """Add video to upload queue."""
        if scheduled_time is None:
            scheduled_time = self._get_next_optimal_time()
        
        self.upload_queue.append({
            'video_path': video_path,
            'metadata': metadata,
            'scheduled_time': scheduled_time
        })
        
        logger.info(f"Queued upload: {metadata.title} for {scheduled_time}")
    
    def _get_next_optimal_time(self) -> datetime:
        """Get the next optimal upload time."""
        now = datetime.utcnow()
        
        for hour in self.OPTIMAL_HOURS:
            candidate = now.replace(hour=hour, minute=0, second=0, microsecond=0)
            if candidate > now:
                return candidate
        
        # Next day
        tomorrow = now + timedelta(days=1)
        return tomorrow.replace(hour=self.OPTIMAL_HOURS[0], minute=0, second=0, microsecond=0)
    
    def process_queue(self) -> List[UploadResult]:
        """Process pending uploads."""
        results = []
        now = datetime.utcnow()
        
        pending = [u for u in self.upload_queue if u['scheduled_time'] <= now]
        
        for upload in pending:
            result = self.uploader.upload_video(
                upload['video_path'],
                upload['metadata']
            )
            results.append(result)
            
            if result.success:
                self.upload_queue.remove(upload)
        
        return results


def create_youtube_uploader(config) -> YouTubeUploader:
    """Factory function to create YouTube uploader."""
    return YouTubeUploader(config)
