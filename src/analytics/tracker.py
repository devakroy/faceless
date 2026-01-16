"""
Analytics and Growth Tracking Module
Tracks video performance, channel growth, and provides insights.
Uses SQLite for local storage (FREE).
"""

import sqlite3
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class VideoStats:
    """Statistics for a single video."""
    video_id: str
    title: str
    views: int
    likes: int
    comments: int
    published_at: str
    niche: str
    duration: int
    thumbnail_ctr: float = 0.0
    avg_view_duration: float = 0.0
    
    def engagement_rate(self) -> float:
        """Calculate engagement rate."""
        if self.views == 0:
            return 0.0
        return ((self.likes + self.comments) / self.views) * 100


@dataclass
class ChannelStats:
    """Channel-level statistics."""
    subscribers: int
    total_views: int
    total_videos: int
    avg_views_per_video: float
    avg_engagement_rate: float
    growth_rate: float  # Subscriber growth rate
    best_performing_niche: str
    best_upload_time: str


@dataclass
class GrowthInsight:
    """Insight for channel growth."""
    category: str  # content, timing, engagement, seo
    insight: str
    action: str
    priority: str  # high, medium, low


class AnalyticsDatabase:
    """SQLite database for storing analytics data."""
    
    def __init__(self, db_path: str = "data/analytics.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """Initialize database tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Videos table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS videos (
                video_id TEXT PRIMARY KEY,
                title TEXT,
                description TEXT,
                niche TEXT,
                duration INTEGER,
                published_at TEXT,
                upload_time TEXT,
                thumbnail_path TEXT,
                script_hash TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Video stats table (historical)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS video_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id TEXT,
                views INTEGER,
                likes INTEGER,
                comments INTEGER,
                recorded_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (video_id) REFERENCES videos(video_id)
            )
        ''')
        
        # Channel stats table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS channel_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subscribers INTEGER,
                total_views INTEGER,
                total_videos INTEGER,
                recorded_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Content performance table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS content_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                niche TEXT,
                avg_views REAL,
                avg_engagement REAL,
                video_count INTEGER,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_video(self, video_id: str, title: str, niche: str, 
                  duration: int, published_at: str, **kwargs):
        """Add a new video to the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO videos 
            (video_id, title, niche, duration, published_at, upload_time, description, thumbnail_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            video_id, title, niche, duration, published_at,
            datetime.now().strftime("%H:%M"),
            kwargs.get('description', ''),
            kwargs.get('thumbnail_path', '')
        ))
        
        conn.commit()
        conn.close()
    
    def update_video_stats(self, video_id: str, views: int, likes: int, comments: int):
        """Update stats for a video."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO video_stats (video_id, views, likes, comments)
            VALUES (?, ?, ?, ?)
        ''', (video_id, views, likes, comments))
        
        conn.commit()
        conn.close()
    
    def update_channel_stats(self, subscribers: int, total_views: int, total_videos: int):
        """Update channel statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO channel_stats (subscribers, total_views, total_videos)
            VALUES (?, ?, ?)
        ''', (subscribers, total_views, total_videos))
        
        conn.commit()
        conn.close()
    
    def get_video_stats(self, video_id: str) -> Optional[VideoStats]:
        """Get latest stats for a video."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT v.video_id, v.title, vs.views, vs.likes, vs.comments, 
                   v.published_at, v.niche, v.duration
            FROM videos v
            LEFT JOIN video_stats vs ON v.video_id = vs.video_id
            WHERE v.video_id = ?
            ORDER BY vs.recorded_at DESC
            LIMIT 1
        ''', (video_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return VideoStats(
                video_id=row[0],
                title=row[1],
                views=row[2] or 0,
                likes=row[3] or 0,
                comments=row[4] or 0,
                published_at=row[5],
                niche=row[6],
                duration=row[7]
            )
        return None
    
    def get_all_videos(self, limit: int = 100) -> List[Dict]:
        """Get all videos with their latest stats."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT v.video_id, v.title, v.niche, v.published_at,
                   COALESCE(vs.views, 0) as views,
                   COALESCE(vs.likes, 0) as likes,
                   COALESCE(vs.comments, 0) as comments
            FROM videos v
            LEFT JOIN (
                SELECT video_id, views, likes, comments,
                       ROW_NUMBER() OVER (PARTITION BY video_id ORDER BY recorded_at DESC) as rn
                FROM video_stats
            ) vs ON v.video_id = vs.video_id AND vs.rn = 1
            ORDER BY v.published_at DESC
            LIMIT ?
        ''', (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [
            {
                'video_id': row[0],
                'title': row[1],
                'niche': row[2],
                'published_at': row[3],
                'views': row[4],
                'likes': row[5],
                'comments': row[6]
            }
            for row in rows
        ]
    
    def get_niche_performance(self) -> Dict[str, Dict]:
        """Get performance metrics by niche."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT v.niche, 
                   COUNT(DISTINCT v.video_id) as video_count,
                   AVG(vs.views) as avg_views,
                   AVG(vs.likes) as avg_likes,
                   AVG(vs.comments) as avg_comments
            FROM videos v
            LEFT JOIN video_stats vs ON v.video_id = vs.video_id
            GROUP BY v.niche
        ''')
        
        rows = cursor.fetchall()
        conn.close()
        
        return {
            row[0]: {
                'video_count': row[1],
                'avg_views': row[2] or 0,
                'avg_likes': row[3] or 0,
                'avg_comments': row[4] or 0,
                'avg_engagement': ((row[3] or 0) + (row[4] or 0)) / max(row[2] or 1, 1) * 100
            }
            for row in rows if row[0]
        }
    
    def get_best_upload_times(self) -> Dict[str, float]:
        """Analyze best upload times based on performance."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT v.upload_time, AVG(vs.views) as avg_views
            FROM videos v
            JOIN video_stats vs ON v.video_id = vs.video_id
            GROUP BY v.upload_time
            ORDER BY avg_views DESC
        ''')
        
        rows = cursor.fetchall()
        conn.close()
        
        return {row[0]: row[1] for row in rows if row[0]}
    
    def get_growth_trend(self, days: int = 30) -> List[Dict]:
        """Get subscriber growth trend."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        
        cursor.execute('''
            SELECT DATE(recorded_at) as date, subscribers, total_views
            FROM channel_stats
            WHERE recorded_at >= ?
            ORDER BY recorded_at
        ''', (cutoff,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [
            {'date': row[0], 'subscribers': row[1], 'views': row[2]}
            for row in rows
        ]


class GrowthAnalyzer:
    """Analyzes channel data and provides growth insights."""
    
    def __init__(self, database: AnalyticsDatabase):
        self.db = database
    
    def analyze(self) -> List[GrowthInsight]:
        """Perform comprehensive growth analysis."""
        insights = []
        
        # Analyze niche performance
        insights.extend(self._analyze_niche_performance())
        
        # Analyze upload timing
        insights.extend(self._analyze_upload_timing())
        
        # Analyze engagement patterns
        insights.extend(self._analyze_engagement())
        
        # Sort by priority
        priority_order = {'high': 0, 'medium': 1, 'low': 2}
        insights.sort(key=lambda x: priority_order.get(x.priority, 3))
        
        return insights
    
    def _analyze_niche_performance(self) -> List[GrowthInsight]:
        """Analyze which niches perform best."""
        insights = []
        niche_data = self.db.get_niche_performance()
        
        if not niche_data:
            return insights
        
        # Find best performing niche
        best_niche = max(niche_data.items(), key=lambda x: x[1]['avg_views'])
        worst_niche = min(niche_data.items(), key=lambda x: x[1]['avg_views'])
        
        if best_niche[1]['avg_views'] > worst_niche[1]['avg_views'] * 1.5:
            insights.append(GrowthInsight(
                category="content",
                insight=f"'{best_niche[0]}' content performs {best_niche[1]['avg_views']/max(worst_niche[1]['avg_views'], 1):.1f}x better than '{worst_niche[0]}'",
                action=f"Focus more on {best_niche[0]} content and reduce {worst_niche[0]} content",
                priority="high"
            ))
        
        return insights
    
    def _analyze_upload_timing(self) -> List[GrowthInsight]:
        """Analyze best upload times."""
        insights = []
        time_data = self.db.get_best_upload_times()
        
        if not time_data:
            return insights
        
        best_time = max(time_data.items(), key=lambda x: x[1])
        
        insights.append(GrowthInsight(
            category="timing",
            insight=f"Videos uploaded at {best_time[0]} get the most views",
            action=f"Schedule uploads around {best_time[0]} for maximum reach",
            priority="medium"
        ))
        
        return insights
    
    def _analyze_engagement(self) -> List[GrowthInsight]:
        """Analyze engagement patterns."""
        insights = []
        videos = self.db.get_all_videos(limit=50)
        
        if not videos:
            return insights
        
        # Calculate average engagement
        total_engagement = 0
        for video in videos:
            if video['views'] > 0:
                engagement = (video['likes'] + video['comments']) / video['views'] * 100
                total_engagement += engagement
        
        avg_engagement = total_engagement / len(videos) if videos else 0
        
        if avg_engagement < 5:
            insights.append(GrowthInsight(
                category="engagement",
                insight=f"Average engagement rate is {avg_engagement:.1f}%, which is below optimal",
                action="Add stronger calls-to-action and ask questions in videos",
                priority="high"
            ))
        elif avg_engagement > 10:
            insights.append(GrowthInsight(
                category="engagement",
                insight=f"Great engagement rate of {avg_engagement:.1f}%!",
                action="Keep up the current content strategy",
                priority="low"
            ))
        
        return insights
    
    def get_recommendations(self) -> Dict[str, Any]:
        """Get actionable recommendations for growth."""
        niche_data = self.db.get_niche_performance()
        time_data = self.db.get_best_upload_times()
        
        recommendations = {
            'best_niche': None,
            'best_upload_time': None,
            'content_mix': {},
            'posting_schedule': []
        }
        
        # Best niche
        if niche_data:
            best = max(niche_data.items(), key=lambda x: x[1]['avg_views'])
            recommendations['best_niche'] = best[0]
            
            # Recommended content mix
            total_views = sum(n['avg_views'] for n in niche_data.values())
            for niche, data in niche_data.items():
                recommendations['content_mix'][niche] = round(
                    data['avg_views'] / total_views * 100, 1
                ) if total_views > 0 else 0
        
        # Best upload time
        if time_data:
            best_time = max(time_data.items(), key=lambda x: x[1])
            recommendations['best_upload_time'] = best_time[0]
        
        return recommendations


class AnalyticsTracker:
    """Main analytics tracker that coordinates all tracking."""
    
    def __init__(self, config):
        self.config = config
        self.db = AnalyticsDatabase(f"{config.storage.output_dir}/analytics.db")
        self.analyzer = GrowthAnalyzer(self.db)
    
    def track_video(self, video_id: str, title: str, niche: str,
                    duration: int, **kwargs):
        """Track a new video upload."""
        self.db.add_video(
            video_id=video_id,
            title=title,
            niche=niche,
            duration=duration,
            published_at=datetime.now().isoformat(),
            **kwargs
        )
        logger.info(f"Tracking video: {video_id}")
    
    def update_stats(self, video_id: str, views: int, likes: int, comments: int):
        """Update video statistics."""
        self.db.update_video_stats(video_id, views, likes, comments)
    
    def update_channel_stats(self, subscribers: int, total_views: int, total_videos: int):
        """Update channel statistics."""
        self.db.update_channel_stats(subscribers, total_views, total_videos)
    
    def get_insights(self) -> List[GrowthInsight]:
        """Get growth insights."""
        return self.analyzer.analyze()
    
    def get_recommendations(self) -> Dict[str, Any]:
        """Get content recommendations."""
        return self.analyzer.get_recommendations()
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for dashboard display."""
        videos = self.db.get_all_videos(limit=10)
        niche_performance = self.db.get_niche_performance()
        growth_trend = self.db.get_growth_trend(days=30)
        insights = self.analyzer.analyze()
        
        return {
            'recent_videos': videos,
            'niche_performance': niche_performance,
            'growth_trend': growth_trend,
            'insights': [asdict(i) for i in insights],
            'recommendations': self.analyzer.get_recommendations()
        }


def create_analytics_tracker(config) -> AnalyticsTracker:
    """Factory function to create analytics tracker."""
    return AnalyticsTracker(config)
