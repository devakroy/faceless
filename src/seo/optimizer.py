"""
SEO Optimization Module for YouTube Videos
Handles title optimization, hashtags, descriptions, and thumbnail generation.
All features are FREE and run locally.
"""

import re
import random
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import logging
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import textwrap

logger = logging.getLogger(__name__)


@dataclass
class SEOResult:
    """Result of SEO optimization."""
    title: str
    description: str
    tags: List[str]
    hashtags: List[str]
    thumbnail_path: Optional[str] = None


class TitleOptimizer:
    """Optimizes video titles for maximum CTR."""
    
    # Power words that increase CTR
    POWER_WORDS = [
        "SECRET", "SHOCKING", "INSANE", "MIND-BLOWING", "INCREDIBLE",
        "AMAZING", "UNBELIEVABLE", "CRAZY", "EPIC", "ULTIMATE",
        "HIDDEN", "REVEALED", "TRUTH", "EXPOSED", "BANNED"
    ]
    
    # Emotional triggers
    EMOTIONAL_TRIGGERS = [
        "You Won't Believe", "This Changed Everything", "Nobody Knows",
        "The Real Reason", "What They Don't Tell You", "Finally Revealed",
        "Stop Doing This", "Why You're Wrong About", "The Harsh Truth"
    ]
    
    # Number patterns that work
    NUMBER_PATTERNS = [
        "3 Things", "5 Secrets", "7 Ways", "10 Reasons", "The #1"
    ]
    
    def optimize(self, title: str, niche: str) -> str:
        """Optimize a title for better CTR."""
        # Clean the title
        title = title.strip()
        
        # Ensure it's not too long (max 100 chars, but 60 is optimal)
        if len(title) > 60:
            title = title[:57] + "..."
        
        # Add emoji if not present (increases CTR)
        if not self._has_emoji(title):
            emoji = self._get_niche_emoji(niche)
            if len(title) + len(emoji) + 1 <= 60:
                title = f"{emoji} {title}"
        
        return title
    
    def _has_emoji(self, text: str) -> bool:
        """Check if text contains emoji."""
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"
            "\U0001F300-\U0001F5FF"
            "\U0001F680-\U0001F6FF"
            "\U0001F1E0-\U0001F1FF"
            "]+",
            flags=re.UNICODE
        )
        return bool(emoji_pattern.search(text))
    
    def _get_niche_emoji(self, niche: str) -> str:
        """Get appropriate emoji for niche."""
        emoji_map = {
            "motivation": "🔥",
            "facts": "🤯",
            "stories": "😱",
            "tech": "🤖",
            "finance": "💰",
            "health": "💪",
            "mystery": "👀"
        }
        return emoji_map.get(niche, "⚡")
    
    def generate_variations(self, title: str, count: int = 5) -> List[str]:
        """Generate title variations for A/B testing."""
        variations = [title]
        
        # Add power word variation
        power_word = random.choice(self.POWER_WORDS)
        variations.append(f"{power_word}: {title}")
        
        # Add question variation
        variations.append(f"Why {title}?")
        
        # Add number variation
        number = random.choice(self.NUMBER_PATTERNS)
        variations.append(f"{number} About {title}")
        
        # Add emotional trigger
        trigger = random.choice(self.EMOTIONAL_TRIGGERS)
        variations.append(f"{trigger}: {title}")
        
        return variations[:count]


class HashtagGenerator:
    """Generates optimized hashtags for YouTube Shorts."""
    
    # Trending hashtags by niche
    NICHE_HASHTAGS = {
        "motivation": [
            "#motivation", "#success", "#mindset", "#grindset", "#hustle",
            "#entrepreneur", "#goals", "#inspiration", "#motivational", "#growth"
        ],
        "facts": [
            "#facts", "#didyouknow", "#science", "#learning", "#education",
            "#interesting", "#knowledge", "#funfacts", "#amazingfacts", "#truth"
        ],
        "stories": [
            "#storytime", "#story", "#drama", "#plot", "#revenge",
            "#karma", "#justice", "#tales", "#narrative", "#storytelling"
        ],
        "tech": [
            "#tech", "#technology", "#ai", "#coding", "#programming",
            "#gadgets", "#innovation", "#future", "#digital", "#software"
        ],
        "finance": [
            "#money", "#finance", "#investing", "#wealth", "#rich",
            "#millionaire", "#passive", "#income", "#financial", "#stocks"
        ],
        "health": [
            "#health", "#fitness", "#wellness", "#healthy", "#workout",
            "#gym", "#nutrition", "#lifestyle", "#selfcare", "#mindfulness"
        ],
        "mystery": [
            "#mystery", "#creepy", "#scary", "#paranormal", "#unsolved",
            "#strange", "#weird", "#unexplained", "#horror", "#dark"
        ]
    }
    
    # Universal viral hashtags
    VIRAL_HASHTAGS = [
        "#shorts", "#viral", "#trending", "#fyp", "#foryou",
        "#explore", "#reels", "#tiktok", "#youtube", "#youtubeshorts"
    ]
    
    def generate(self, niche: str, keywords: List[str], count: int = 15) -> List[str]:
        """Generate optimized hashtags."""
        hashtags = []
        
        # Add viral hashtags (always include these)
        hashtags.extend(self.VIRAL_HASHTAGS[:5])
        
        # Add niche-specific hashtags
        niche_tags = self.NICHE_HASHTAGS.get(niche, [])
        hashtags.extend(random.sample(niche_tags, min(5, len(niche_tags))))
        
        # Add keyword-based hashtags
        for keyword in keywords[:5]:
            tag = f"#{keyword.lower().replace(' ', '')}"
            if tag not in hashtags:
                hashtags.append(tag)
        
        # Ensure uniqueness and limit
        hashtags = list(dict.fromkeys(hashtags))[:count]
        
        return hashtags


class DescriptionOptimizer:
    """Optimizes video descriptions for SEO."""
    
    DESCRIPTION_TEMPLATE = """{title}

{hook}

{hashtags}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔔 Turn on notifications to never miss a video!
👍 Like this video if you found it valuable
💬 Comment your thoughts below
📱 Share with someone who needs to see this

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Keywords: {keywords}

#shorts #viral #trending #fyp #youtube"""
    
    def optimize(self, title: str, hook: str, keywords: List[str], 
                 hashtags: List[str]) -> str:
        """Generate optimized description."""
        return self.DESCRIPTION_TEMPLATE.format(
            title=title,
            hook=hook,
            hashtags=" ".join(hashtags[:10]),
            keywords=", ".join(keywords[:10])
        )


class ThumbnailGenerator:
    """Generates eye-catching thumbnails for videos."""
    
    # Color schemes that work
    COLOR_SCHEMES = {
        "motivation": {
            "bg": (255, 87, 51),  # Orange-red
            "text": (255, 255, 255),
            "accent": (255, 215, 0)
        },
        "facts": {
            "bg": (0, 150, 199),  # Blue
            "text": (255, 255, 255),
            "accent": (255, 255, 0)
        },
        "stories": {
            "bg": (75, 0, 130),  # Indigo
            "text": (255, 255, 255),
            "accent": (255, 105, 180)
        },
        "tech": {
            "bg": (0, 0, 0),  # Black
            "text": (0, 255, 255),
            "accent": (255, 0, 255)
        },
        "finance": {
            "bg": (0, 100, 0),  # Green
            "text": (255, 215, 0),
            "accent": (255, 255, 255)
        },
        "health": {
            "bg": (34, 139, 34),  # Forest green
            "text": (255, 255, 255),
            "accent": (255, 215, 0)
        },
        "mystery": {
            "bg": (25, 25, 25),  # Dark
            "text": (255, 0, 0),
            "accent": (255, 255, 255)
        }
    }
    
    def __init__(self, output_dir: str = "output/thumbnails"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fonts_dir = Path("assets/fonts")
    
    def _get_font(self, size: int) -> ImageFont.FreeTypeFont:
        """Get font for thumbnail text."""
        font_paths = [
            self.fonts_dir / "Montserrat-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
        ]
        
        for font_path in font_paths:
            if Path(font_path).exists():
                return ImageFont.truetype(str(font_path), size)
        
        return ImageFont.load_default()
    
    def generate(self, title: str, niche: str, 
                 background_image: Optional[str] = None,
                 output_filename: Optional[str] = None) -> str:
        """
        Generate a thumbnail for the video.
        
        Args:
            title: Video title (will be shortened for thumbnail)
            niche: Content niche for color scheme
            background_image: Optional background image path
            output_filename: Output filename
        
        Returns:
            Path to generated thumbnail
        """
        # Thumbnail size (YouTube recommended: 1280x720)
        width, height = 1280, 720
        
        # Get color scheme
        colors = self.COLOR_SCHEMES.get(niche, self.COLOR_SCHEMES["motivation"])
        
        # Create base image
        if background_image and Path(background_image).exists():
            img = Image.open(background_image)
            img = img.resize((width, height), Image.Resampling.LANCZOS)
            # Darken background
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(0.4)
        else:
            # Create gradient background
            img = self._create_gradient(width, height, colors["bg"])
        
        draw = ImageDraw.Draw(img)
        
        # Prepare text (shorten for thumbnail)
        text = self._prepare_thumbnail_text(title)
        
        # Draw text with outline
        self._draw_text_with_outline(
            draw, text, width, height,
            colors["text"], colors["accent"]
        )
        
        # Add visual elements
        self._add_visual_elements(draw, width, height, colors)
        
        # Save thumbnail
        if output_filename is None:
            output_filename = f"thumb_{hash(title) % 10000}.jpg"
        
        output_path = self.output_dir / output_filename
        img.save(output_path, "JPEG", quality=95)
        
        logger.info(f"Generated thumbnail: {output_path}")
        
        return str(output_path)
    
    def _create_gradient(self, width: int, height: int, 
                         base_color: Tuple[int, int, int]) -> Image.Image:
        """Create a gradient background."""
        img = Image.new('RGB', (width, height))
        draw = ImageDraw.Draw(img)
        
        # Create vertical gradient
        for y in range(height):
            ratio = y / height
            r = int(base_color[0] * (1 - ratio * 0.5))
            g = int(base_color[1] * (1 - ratio * 0.5))
            b = int(base_color[2] * (1 - ratio * 0.5))
            draw.line([(0, y), (width, y)], fill=(r, g, b))
        
        return img
    
    def _prepare_thumbnail_text(self, title: str) -> str:
        """Prepare text for thumbnail (short and impactful)."""
        # Remove emojis
        title = re.sub(r'[^\w\s!?.,]', '', title)
        
        # Shorten if needed
        words = title.split()
        if len(words) > 5:
            title = " ".join(words[:5]) + "..."
        
        # Uppercase for impact
        return title.upper()
    
    def _draw_text_with_outline(self, draw: ImageDraw.Draw, text: str,
                                 width: int, height: int,
                                 text_color: Tuple, outline_color: Tuple):
        """Draw text with outline effect."""
        # Wrap text
        wrapped = textwrap.fill(text, width=15)
        lines = wrapped.split('\n')
        
        # Calculate font size based on text length
        font_size = min(120, int(width / max(len(line) for line in lines) * 1.5))
        font = self._get_font(font_size)
        
        # Calculate total text height
        line_height = font_size + 10
        total_height = len(lines) * line_height
        
        # Starting Y position (centered)
        y = (height - total_height) // 2
        
        for line in lines:
            # Get text bounding box
            bbox = draw.textbbox((0, 0), line, font=font)
            text_width = bbox[2] - bbox[0]
            x = (width - text_width) // 2
            
            # Draw outline
            outline_width = 4
            for dx in range(-outline_width, outline_width + 1):
                for dy in range(-outline_width, outline_width + 1):
                    draw.text((x + dx, y + dy), line, font=font, fill=(0, 0, 0))
            
            # Draw main text
            draw.text((x, y), line, font=font, fill=text_color)
            
            y += line_height
    
    def _add_visual_elements(self, draw: ImageDraw.Draw, width: int, height: int,
                             colors: Dict):
        """Add visual elements to make thumbnail pop."""
        # Add corner accents
        accent = colors["accent"]
        
        # Top-left corner
        draw.polygon([(0, 0), (100, 0), (0, 100)], fill=accent)
        
        # Bottom-right corner
        draw.polygon([(width, height), (width - 100, height), (width, height - 100)], fill=accent)
        
        # Add border
        border_width = 8
        draw.rectangle(
            [border_width, border_width, width - border_width, height - border_width],
            outline=accent,
            width=border_width
        )


class SEOOptimizer:
    """Main SEO optimizer that combines all optimization features."""
    
    def __init__(self, config):
        self.config = config
        self.title_optimizer = TitleOptimizer()
        self.hashtag_generator = HashtagGenerator()
        self.description_optimizer = DescriptionOptimizer()
        self.thumbnail_generator = ThumbnailGenerator(
            output_dir=f"{config.storage.output_dir}/thumbnails"
        )
    
    def optimize(self, title: str, hook: str, keywords: List[str],
                 niche: str, background_image: Optional[str] = None) -> SEOResult:
        """
        Perform full SEO optimization.
        
        Args:
            title: Video title
            hook: Video hook/intro
            keywords: Content keywords
            niche: Content niche
            background_image: Optional image for thumbnail
        
        Returns:
            SEOResult with optimized metadata
        """
        # Optimize title
        optimized_title = self.title_optimizer.optimize(title, niche)
        
        # Generate hashtags
        hashtags = self.hashtag_generator.generate(niche, keywords)
        
        # Generate tags (without # prefix)
        tags = [h.replace("#", "") for h in hashtags]
        tags.extend(keywords)
        tags = list(dict.fromkeys(tags))[:30]  # YouTube allows max 500 chars total
        
        # Optimize description
        description = self.description_optimizer.optimize(
            optimized_title, hook, keywords, hashtags
        )
        
        # Generate thumbnail
        thumbnail_path = None
        if self.config.thumbnail.enabled:
            thumbnail_path = self.thumbnail_generator.generate(
                title, niche, background_image
            )
        
        return SEOResult(
            title=optimized_title,
            description=description,
            tags=tags,
            hashtags=hashtags,
            thumbnail_path=thumbnail_path
        )


def create_seo_optimizer(config) -> SEOOptimizer:
    """Factory function to create SEO optimizer."""
    return SEOOptimizer(config)
