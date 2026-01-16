# 🎬 Faceless YouTube Channel Automation

A **100% FREE**, fully automated system for creating and managing a faceless YouTube channel. All processing runs locally on your CPU/GPU - no expensive cloud services required!

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Cost](https://img.shields.io/badge/Cost-$0-brightgreen.svg)

---

## 📋 Table of Contents

1. [Project Overview](#-project-overview)
2. [Architecture & Data Flow](#-architecture--data-flow)
3. [Project Structure](#-project-structure)
4. [Detailed File Documentation](#-detailed-file-documentation)
5. [Setup Instructions](#-setup-instructions)
6. [Configuration Guide](#-configuration-guide)
7. [Usage Guide](#-usage-guide)
8. [API Reference](#-api-reference)
9. [Growth Strategy](#-growth-strategy)
10. [Troubleshooting](#-troubleshooting)
11. [Contributing](#-contributing)

---

## 🎯 Project Overview

### What is this project?

This is an **automated content creation pipeline** for YouTube that:
1. **Generates viral scripts** using AI (local LLMs via Ollama)
2. **Converts scripts to speech** using Text-to-Speech engines
3. **Creates videos** with background footage and animated subtitles
4. **Optimizes for SEO** with titles, descriptions, tags, and thumbnails
5. **Uploads to YouTube** automatically at optimal times
6. **Tracks analytics** and provides growth insights

### Why was it built?

- **Cost**: Traditional video creation tools cost $50-500/month. This is FREE.
- **Automation**: Manual video creation takes 2-4 hours per video. This takes 5 minutes.
- **Scalability**: Create 1 or 100 videos with the same effort.
- **Growth**: Built-in SEO optimization and analytics for organic growth.

### Who is it for?

- Content creators wanting passive income
- Developers learning about video automation
- Anyone interested in AI-powered content creation

---

## 🏗 Architecture & Data Flow

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        FACELESS AUTOMATION SYSTEM                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐               │
│  │   AI Script  │───▶│  TTS Engine  │───▶│    Video     │               │
│  │  Generator   │    │              │    │  Generator   │               │
│  └──────────────┘    └──────────────┘    └──────────────┘               │
│         │                   │                   │                        │
│         │                   │                   │                        │
│         ▼                   ▼                   ▼                        │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐               │
│  │    Stock     │    │     SEO      │    │   YouTube    │               │
│  │    Media     │───▶│  Optimizer   │───▶│   Uploader   │               │
│  └──────────────┘    └──────────────┘    └──────────────┘               │
│                             │                   │                        │
│                             ▼                   ▼                        │
│                      ┌──────────────┐    ┌──────────────┐               │
│                      │  Analytics   │◀───│  Scheduler   │               │
│                      │   Tracker    │    │              │               │
│                      └──────────────┘    └──────────────┘               │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Data Flow (Step by Step)

```
1. SCRIPT GENERATION
   Input:  Niche (e.g., "motivation"), Duration (e.g., 60 seconds)
   Output: VideoScript object with title, hook, body, CTA, keywords
   
2. TEXT-TO-SPEECH
   Input:  Script text
   Output: Audio file (WAV) + word timestamps for subtitles
   
3. MEDIA SOURCING
   Input:  Niche keywords
   Output: List of downloaded stock video/image paths
   
4. VIDEO GENERATION
   Input:  Audio + Background media + Script text
   Output: Final video file (MP4) with animated subtitles
   
5. SEO OPTIMIZATION
   Input:  Title, keywords, niche
   Output: Optimized title, description, tags, hashtags, thumbnail
   
6. YOUTUBE UPLOAD
   Input:  Video file + Metadata
   Output: Video ID, URL, upload status
   
7. ANALYTICS TRACKING
   Input:  Video ID, performance metrics
   Output: Insights, recommendations, growth data
```

---

## 📁 Project Structure

```
faceless/
│
├── 📄 main.py                      # CLI entry point - all user commands
├── 📄 requirements.txt             # Python package dependencies
├── 📄 README.md                    # This documentation file
├── 📄 QUICKSTART.md               # 10-minute setup guide
├── 📄 .gitignore                  # Git ignore rules
│
├── 📁 config/                      # Configuration files
│   ├── 📄 config.yaml             # Main configuration (edit this!)
│   ├── 📄 client_secrets.json     # YouTube API credentials (you create)
│   └── 📄 youtube_credentials.json # Auto-generated after first auth
│
├── 📁 scripts/                     # Utility scripts
│   └── 📄 setup.sh                # Automated setup script
│
├── 📁 src/                         # Source code
│   ├── 📄 __init__.py             # Package marker
│   │
│   ├── 📁 ai/                     # AI Script Generation Module
│   │   ├── 📄 __init__.py
│   │   └── 📄 script_generator.py # Generates viral video scripts
│   │
│   ├── 📁 audio/                  # Text-to-Speech Module
│   │   ├── 📄 __init__.py
│   │   └── 📄 tts_engine.py       # Converts text to speech
│   │
│   ├── 📁 media/                  # Stock Media Module
│   │   ├── 📄 __init__.py
│   │   └── 📄 stock_media.py      # Downloads stock videos/images
│   │
│   ├── 📁 video/                  # Video Generation Module
│   │   ├── 📄 __init__.py
│   │   └── 📄 video_generator.py  # Creates final video with subtitles
│   │
│   ├── 📁 youtube/                # YouTube Upload Module
│   │   ├── 📄 __init__.py
│   │   └── 📄 uploader.py         # Handles YouTube API uploads
│   │
│   ├── 📁 seo/                    # SEO Optimization Module
│   │   ├── 📄 __init__.py
│   │   └── 📄 optimizer.py        # Optimizes titles, tags, thumbnails
│   │
│   ├── 📁 analytics/              # Analytics Module
│   │   ├── 📄 __init__.py
│   │   └── 📄 tracker.py          # Tracks performance, provides insights
│   │
│   ├── 📁 pipeline/               # Main Automation Pipeline
│   │   ├── 📄 __init__.py
│   │   └── 📄 automation.py       # Orchestrates entire workflow
│   │
│   └── 📁 utils/                  # Utility Functions
│       ├── 📄 __init__.py
│       └── 📄 config_loader.py    # Loads and validates configuration
│
├── 📁 output/                      # Generated videos (auto-created)
├── 📁 temp/                        # Temporary files (auto-created)
├── 📁 logs/                        # Log files (auto-created)
├── 📁 cache/                       # Cached media (auto-created)
├── 📁 data/                        # Database files (auto-created)
├── 📁 models/                      # AI model files (auto-created)
└── 📁 assets/                      # Static assets
    ├── 📁 fonts/                  # Custom fonts
    ├── 📁 music/                  # Background music
    └── 📁 media/                  # Local media library
```

---

## 📚 Detailed File Documentation

### Core Files

#### `main.py` - CLI Entry Point
**Purpose**: Provides command-line interface for all operations.

**Commands**:
| Command | Description | Example |
|---------|-------------|---------|
| `create` | Create single video | `python main.py create --niche motivation` |
| `batch N` | Create N videos | `python main.py batch 5` |
| `start` | Start 24/7 automation | `python main.py start` |
| `status` | Check automation status | `python main.py status` |
| `insights` | Get growth insights | `python main.py insights` |
| `setup` | Run setup wizard | `python main.py setup` |
| `test` | Test all components | `python main.py test` |

**Key Functions**:
- Uses `click` library for CLI
- Uses `rich` library for beautiful terminal output
- Initializes and coordinates all modules

---

#### `config/config.yaml` - Main Configuration
**Purpose**: Central configuration file for all settings.

**Sections**:
```yaml
channel:        # Channel settings (name, niche, format)
ai:             # AI provider settings (ollama, groq, huggingface)
tts:            # Text-to-speech settings (provider, voice, speed)
video:          # Video settings (resolution, fps, gpu)
media:          # Stock media API keys
subtitles:      # Subtitle styling
youtube:        # YouTube API settings
seo:            # SEO optimization settings
storage:        # Output directories
scheduler:      # Automation schedule
```

**Important Settings to Configure**:
1. `channel.niche` - Your content niche
2. `media.pexels_api_key` - For stock videos (optional but recommended)
3. `youtube.client_secrets_file` - Path to YouTube credentials

---

### Source Code Modules

#### `src/ai/script_generator.py` - AI Script Generation
**Purpose**: Generates viral video scripts using AI.

**Classes**:
| Class | Purpose |
|-------|---------|
| `VideoScript` | Data class holding script components |
| `ScriptPromptTemplates` | Templates for different niches |
| `OllamaProvider` | Local LLM via Ollama (FREE, unlimited) |
| `GroqProvider` | Groq API (FREE tier: 30 req/min) |
| `HuggingFaceProvider` | HuggingFace API (FREE tier) |
| `ScriptGenerator` | Main class that generates scripts |

**Key Functions**:
```python
# Generate a single script
script = generator.generate_script(niche="motivation", duration=60)

# Generate multiple scripts
scripts = generator.generate_batch(niche="facts", count=5)
```

**Output (VideoScript)**:
```python
VideoScript(
    title="The One Habit That Changed Everything",
    hook="Nobody tells you this about success...",
    body="The main content of the video...",
    call_to_action="Follow for more tips!",
    full_script="Complete script text...",
    keywords=["success", "habits", "motivation"],
    hashtags=["#shorts", "#motivation", "#success"],
    description="YouTube description...",
    estimated_duration=60,
    niche="motivation"
)
```

---

#### `src/audio/tts_engine.py` - Text-to-Speech
**Purpose**: Converts script text to natural-sounding speech.

**Classes**:
| Class | Purpose |
|-------|---------|
| `AudioResult` | Data class with audio path and duration |
| `PiperTTS` | Fast local TTS (requires model download) |
| `EdgeTTS` | Microsoft's FREE TTS (no setup needed!) |
| `CoquiTTS` | High-quality neural TTS (slower) |
| `TTSEngine` | Main class that manages TTS |

**Key Functions**:
```python
# Generate speech
result = engine.generate_speech(text="Hello world", output_path="speech.wav")

# Generate with word timestamps (for animated subtitles)
result = engine.generate_with_timestamps(text="Hello world")
```

**Output (AudioResult)**:
```python
AudioResult(
    audio_path="/path/to/audio.wav",
    duration=5.5,  # seconds
    sample_rate=22050,
    word_timestamps=[("Hello", 0.0, 0.5), ("world", 0.6, 1.0)]
)
```

**Recommended Provider**: `edge_tts` - No setup, high quality, unlimited use!

---

#### `src/media/stock_media.py` - Stock Media Sourcing
**Purpose**: Downloads royalty-free videos and images for backgrounds.

**Classes**:
| Class | Purpose |
|-------|---------|
| `MediaItem` | Data class for media metadata |
| `PexelsProvider` | Pexels API (200 req/hour FREE) |
| `PixabayProvider` | Pixabay API (100 req/min FREE) |
| `LocalMediaProvider` | Uses local media library |
| `StockMediaManager` | Manages all providers |

**Key Functions**:
```python
# Get background videos
videos = manager.get_background_videos(niche="motivation", count=3)

# Get background images
images = manager.get_background_images(niche="tech", count=5)
```

**Niche Keywords** (automatically selected):
```python
NICHE_KEYWORDS = {
    "motivation": ["success", "mountain top", "sunrise", "running"],
    "facts": ["science", "space", "nature", "technology"],
    "stories": ["dramatic", "cinematic", "dark", "mystery"],
    # ... more niches
}
```

---

#### `src/video/video_generator.py` - Video Generation
**Purpose**: Creates final video with background, audio, and animated subtitles.

**Classes**:
| Class | Purpose |
|-------|---------|
| `SubtitleStyle` | Subtitle styling configuration |
| `VideoSpec` | Video specifications (resolution, fps) |
| `SubtitleGenerator` | Creates animated subtitle clips |
| `VideoCompositor` | Combines all elements |
| `VideoGenerator` | Main class for video creation |

**Key Functions**:
```python
# Generate video
video_path = generator.generate(
    script_text="Your script here",
    audio_path="/path/to/audio.wav",
    background_videos=["/path/to/bg1.mp4", "/path/to/bg2.mp4"],
    word_timestamps=[("word", 0.0, 0.5), ...]
)
```

**Video Specifications**:
| Format | Resolution | Aspect Ratio | Duration |
|--------|------------|--------------|----------|
| Shorts | 1080x1920 | 9:16 | ≤60 sec |
| Long | 1920x1080 | 16:9 | Any |

**Subtitle Features**:
- Word-by-word animation
- Customizable fonts, colors, positions
- Stroke/outline for readability
- Fade in/out effects

---

#### `src/youtube/uploader.py` - YouTube Upload
**Purpose**: Handles YouTube API authentication and video uploads.

**Classes**:
| Class | Purpose |
|-------|---------|
| `VideoMetadata` | Video metadata for upload |
| `UploadResult` | Result of upload operation |
| `YouTubeAuthenticator` | Handles OAuth2 authentication |
| `YouTubeUploader` | Main upload class |
| `UploadScheduler` | Schedules uploads at optimal times |

**Key Functions**:
```python
# Upload video
result = uploader.upload_video(
    video_path="/path/to/video.mp4",
    metadata=VideoMetadata(
        title="My Video Title",
        description="Video description...",
        tags=["tag1", "tag2"],
        privacy_status="public"
    )
)

# Get channel info
info = uploader.get_channel_info()
```

**YouTube API Quota**:
- Daily limit: 10,000 units
- Upload cost: ~1,600 units
- You can upload ~6 videos/day

**First-Time Setup**:
1. Browser opens for Google OAuth
2. Grant YouTube permissions
3. Credentials saved for future use

---

#### `src/seo/optimizer.py` - SEO Optimization
**Purpose**: Optimizes video metadata for discoverability and generates thumbnails.

**Classes**:
| Class | Purpose |
|-------|---------|
| `SEOResult` | Optimized metadata result |
| `TitleOptimizer` | Optimizes titles with power words |
| `HashtagGenerator` | Generates trending hashtags |
| `DescriptionOptimizer` | Creates SEO-friendly descriptions |
| `ThumbnailGenerator` | Creates eye-catching thumbnails |
| `SEOOptimizer` | Main SEO class |

**Key Functions**:
```python
# Full SEO optimization
result = optimizer.optimize(
    title="Original Title",
    hook="Video hook",
    keywords=["keyword1", "keyword2"],
    niche="motivation"
)
```

**Output (SEOResult)**:
```python
SEOResult(
    title="🔥 SHOCKING: Original Title",
    description="Optimized description with hashtags...",
    tags=["motivation", "success", "viral"],
    hashtags=["#shorts", "#viral", "#motivation"],
    thumbnail_path="/path/to/thumbnail.jpg"
)
```

**Thumbnail Features**:
- Auto-generated from title
- Niche-specific color schemes
- Bold text with outlines
- 1280x720 resolution (YouTube recommended)

---

#### `src/analytics/tracker.py` - Analytics Tracking
**Purpose**: Tracks video performance and provides growth insights.

**Classes**:
| Class | Purpose |
|-------|---------|
| `VideoStats` | Statistics for a video |
| `ChannelStats` | Channel-level statistics |
| `GrowthInsight` | Actionable growth insight |
| `AnalyticsDatabase` | SQLite database for storage |
| `GrowthAnalyzer` | Analyzes data for insights |
| `AnalyticsTracker` | Main analytics class |

**Key Functions**:
```python
# Track a video
tracker.track_video(video_id="abc123", title="My Video", niche="motivation")

# Get insights
insights = tracker.get_insights()

# Get recommendations
recommendations = tracker.get_recommendations()
```

**Insights Provided**:
- Best performing niche
- Optimal upload times
- Engagement rate analysis
- Content mix recommendations

---

#### `src/pipeline/automation.py` - Main Pipeline
**Purpose**: Orchestrates the entire video creation workflow.

**Classes**:
| Class | Purpose |
|-------|---------|
| `PipelineResult` | Result of pipeline run |
| `ContentPipeline` | Single video creation pipeline |
| `AutomationScheduler` | 24/7 scheduling |
| `FacelessAutomation` | Main automation interface |

**Key Functions**:
```python
# Create automation instance
automation = FacelessAutomation()

# Create single video
result = automation.create_video(niche="motivation")

# Create batch
results = automation.create_batch(count=5)

# Start 24/7 automation
automation.start_automation()

# Get status
status = automation.get_status()
```

**Pipeline Steps**:
1. Generate script (AI)
2. Generate audio (TTS)
3. Fetch background media
4. Generate video
5. Optimize SEO
6. Upload to YouTube
7. Track analytics

---

#### `src/utils/config_loader.py` - Configuration
**Purpose**: Loads and validates configuration from YAML.

**Classes**:
| Class | Purpose |
|-------|---------|
| `Config` | Main configuration dataclass |
| `AIConfig`, `TTSConfig`, etc. | Section-specific configs |
| `ConfigLoader` | Loads and parses YAML |

**Key Functions**:
```python
# Get global config
config = get_config()

# Access settings
niche = config.channel.niche
ai_provider = config.ai.provider
```

---

## 🔧 Setup Instructions

### Prerequisites

| Requirement | Version | Purpose |
|-------------|---------|---------|
| Python | 3.9+ | Runtime |
| FFmpeg | Any | Video processing |
| Ollama | Latest | Local AI |

### Step-by-Step Setup

#### 1. Clone Repository
```bash
git clone https://github.com/yourusername/faceless.git
cd faceless
```

#### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate     # Windows
```

#### 3. Install Python Dependencies
```bash
pip install -r requirements.txt
```

#### 4. Install FFmpeg
```bash
# Ubuntu/Debian
sudo apt update && sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
# Add to PATH
```

#### 5. Install Ollama (Local AI)
```bash
# Install
curl -fsSL https://ollama.com/install.sh | sh

# Pull model
ollama pull llama3.2

# Start server (keep running)
ollama serve
```

#### 6. Setup YouTube API
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create new project
3. Enable "YouTube Data API v3"
4. Create OAuth 2.0 credentials (Desktop application)
5. Download JSON → Save as `config/client_secrets.json`

#### 7. Configure
Edit `config/config.yaml`:
```yaml
channel:
  name: "Your Channel Name"
  niche: "motivation"

media:
  pexels_api_key: "your-key"  # Get from pexels.com/api
```

#### 8. Test
```bash
python main.py test
```

#### 9. Create First Video
```bash
python main.py create
```

---

## ⚙️ Configuration Guide

### Niche Options

| Niche | Description | Best For |
|-------|-------------|----------|
| `motivation` | Inspirational content | Self-improvement audience |
| `facts` | Interesting facts | Curious viewers |
| `stories` | Engaging narratives | Entertainment seekers |
| `tech` | Technology content | Tech enthusiasts |
| `finance` | Money & investing | Financial audience |
| `health` | Wellness content | Health-conscious viewers |
| `mystery` | Mysterious content | Mystery lovers |

### AI Provider Comparison

| Provider | Setup | Cost | Speed | Quality |
|----------|-------|------|-------|---------|
| Ollama | Install locally | FREE | Medium | ⭐⭐⭐⭐⭐ |
| Groq | API key | FREE tier | Fast | ⭐⭐⭐⭐⭐ |
| HuggingFace | API token | FREE tier | Slow | ⭐⭐⭐⭐ |

### TTS Provider Comparison

| Provider | Setup | Cost | Speed | Quality |
|----------|-------|------|-------|---------|
| Edge TTS | None! | FREE | Fast | ⭐⭐⭐⭐⭐ |
| Piper | Download models | FREE | Very Fast | ⭐⭐⭐⭐ |
| Coqui | pip install | FREE | Slow | ⭐⭐⭐⭐⭐ |

**Recommendation**: Use `edge_tts` - no setup required, high quality!

---

## 🚀 Usage Guide

### Create Single Video
```bash
python main.py create
python main.py create --niche facts
```

### Create Multiple Videos
```bash
python main.py batch 5
python main.py batch 10 --niches motivation facts stories
```

### Start 24/7 Automation
```bash
python main.py start
# Press Ctrl+C to stop
```

### Check Status
```bash
python main.py status
```

### Get Growth Insights
```bash
python main.py insights
```

### Test Components
```bash
python main.py test
```

---

## 📖 API Reference

### FacelessAutomation

```python
from src.pipeline.automation import FacelessAutomation

# Initialize
automation = FacelessAutomation(config_path="config/config.yaml")

# Create video
result = automation.create_video(niche="motivation")
# Returns: PipelineResult(success=True, video_id="abc123", ...)

# Create batch
results = automation.create_batch(count=5, niches=["motivation", "facts"])
# Returns: List[PipelineResult]

# Start automation
automation.start_automation()

# Stop automation
automation.stop_automation()

# Get status
status = automation.get_status()
# Returns: {"scheduler": {...}, "channel": {...}, "analytics": {...}}

# Get insights
insights = automation.get_insights()
# Returns: [{"category": "content", "insight": "...", "action": "..."}]
```

### ScriptGenerator

```python
from src.ai.script_generator import create_script_generator

generator = create_script_generator(config)

# Generate script
script = generator.generate_script(niche="motivation", duration=60)
# Returns: VideoScript object

# Generate batch
scripts = generator.generate_batch(niche="facts", count=5)
# Returns: List[VideoScript]
```

### TTSEngine

```python
from src.audio.tts_engine import create_tts_engine

engine = create_tts_engine(config)

# Generate speech
result = engine.generate_speech(text="Hello world")
# Returns: AudioResult(audio_path="...", duration=1.5)

# With timestamps
result = engine.generate_with_timestamps(text="Hello world")
# Returns: AudioResult with word_timestamps
```

### VideoGenerator

```python
from src.video.video_generator import create_video_generator

generator = create_video_generator(config)

# Generate video
path = generator.generate(
    script_text="Your script",
    audio_path="audio.wav",
    background_videos=["bg1.mp4", "bg2.mp4"]
)
# Returns: "/path/to/output.mp4"
```

### YouTubeUploader

```python
from src.youtube.uploader import create_youtube_uploader, VideoMetadata

uploader = create_youtube_uploader(config)

# Upload
result = uploader.upload_video(
    video_path="video.mp4",
    metadata=VideoMetadata(title="Title", description="Desc", tags=["tag"])
)
# Returns: UploadResult(success=True, video_id="abc123")
```

---

## 📈 Growth Strategy

### Viral Hook Formulas

| Type | Formula | Example |
|------|---------|---------|
| Question | "Did you know...?" | "Did you know your brain does this?" |
| Shocking | "This will blow your mind..." | "This fact will blow your mind..." |
| Story | "A man once..." | "A man once tried this and..." |
| Secret | "Nobody tells you..." | "Nobody tells you this about success..." |
| Challenge | "Try this..." | "Try this for 30 days..." |

### Optimal Upload Times

| Time (UTC) | Audience | Engagement |
|------------|----------|------------|
| 09:00 | Morning viewers | High |
| 15:00 | Afternoon break | Medium |
| 21:00 | Evening relaxation | Highest |

### Monetization Path

| Requirement | Target | Estimated Time |
|-------------|--------|----------------|
| Subscribers | 1,000 | 3-6 months |
| Watch Hours | 4,000 | 6-12 months |
| OR Shorts Views | 10M (90 days) | 3-6 months |

---

## 🔧 Troubleshooting

### Common Issues

#### Ollama Not Connecting
```bash
# Check if running
curl http://localhost:11434/api/tags

# Start Ollama
ollama serve
```

#### FFmpeg Not Found
```bash
# Verify installation
ffmpeg -version

# Install if missing
sudo apt install ffmpeg  # Linux
brew install ffmpeg      # Mac
```

#### YouTube Upload Fails
1. Check `config/client_secrets.json` exists
2. Delete `config/youtube_credentials.json` and re-authenticate
3. Verify YouTube Data API is enabled in Google Cloud Console
4. Check daily quota (10,000 units)

#### Video Generation Slow
1. Enable GPU in config: `video.use_gpu: true`
2. Reduce resolution for testing
3. Use shorter duration

#### No Stock Videos
1. Add Pexels API key: https://www.pexels.com/api/
2. Add Pixabay API key: https://pixabay.com/api/docs/
3. Add local videos to `assets/media/`

---

## 🤝 Contributing

### How to Contribute

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open Pull Request

### Code Style

- Follow PEP 8
- Add docstrings to all functions
- Include type hints
- Write unit tests for new features

### Areas for Improvement

- [ ] Add more TTS voices
- [ ] Support more video formats
- [ ] Add A/B testing for titles
- [ ] Implement comment automation
- [ ] Add multi-language support

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file.

---

## ⚠️ Disclaimer

- Follow YouTube's Terms of Service
- Don't spam or create misleading content
- Use only royalty-free media
- This tool is for educational purposes

---

## 🙏 Acknowledgments

- [Ollama](https://ollama.com/) - Local LLM runner
- [MoviePy](https://zulko.github.io/moviepy/) - Video editing
- [Edge TTS](https://github.com/rany2/edge-tts) - Microsoft TTS
- [Pexels](https://www.pexels.com/) - Stock videos
- [Pixabay](https://pixabay.com/) - Stock media

---

**Made with ❤️ for content creators**

*Star ⭐ this repo if you find it useful!*
