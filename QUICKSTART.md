# 🚀 Quick Start Guide

Get your faceless YouTube channel running in 10 minutes!

## Step 1: Install Dependencies (2 min)

```bash
# Clone the repo
cd faceless

# Run setup script
chmod +x scripts/setup.sh
./scripts/setup.sh

# Or manually:
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Step 2: Install Ollama (2 min)

```bash
# Install Ollama (FREE local AI)
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model
ollama pull llama3.2

# Start Ollama (keep this running)
ollama serve
```

## Step 3: Configure (2 min)

Edit `config/config.yaml`:

```yaml
channel:
  name: "My Motivation Channel"
  niche: "motivation"  # Choose: motivation, facts, stories, tech, finance, health, mystery

# Optional: Add API keys for better stock videos
media:
  pexels_api_key: "your-key-here"  # Get free at pexels.com/api
```

## Step 4: Setup YouTube (3 min)

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create new project → Enable YouTube Data API v3
3. Create OAuth credentials (Desktop app)
4. Download → Save as `config/client_secrets.json`

## Step 5: Create Your First Video! (1 min)

```bash
# Activate environment
source venv/bin/activate

# Create a video
python main.py create

# First time: Browser opens for YouTube auth
# After that: Fully automated!
```

## 🎉 That's It!

Your first video will be:
- Generated with AI script
- Voiced with natural TTS
- Combined with stock footage
- Uploaded to YouTube
- Optimized for SEO

## Next Steps

```bash
# Create 5 videos at once
python main.py batch 5

# Start 24/7 automation
python main.py start

# Check status
python main.py status

# Get growth insights
python main.py insights
```

## Troubleshooting

**Ollama not working?**
```bash
# Make sure it's running
ollama serve
```

**No stock videos?**
- Add Pexels API key (free): https://www.pexels.com/api/
- Add Pixabay API key (free): https://pixabay.com/api/docs/

**YouTube upload fails?**
- Check `config/client_secrets.json` exists
- Delete `config/youtube_credentials.json` and re-auth

## Cost: $0

Everything is FREE:
- ✅ Ollama - Local AI (unlimited)
- ✅ Edge TTS - Microsoft voices (unlimited)
- ✅ Pexels/Pixabay - Stock media (generous free tiers)
- ✅ YouTube API - 10,000 quota/day (enough for ~6 uploads)

---

**Happy Creating! 🎬**
