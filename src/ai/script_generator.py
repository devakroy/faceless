"""
AI Script Generator for Faceless YouTube Videos
Supports multiple FREE AI providers:
- Ollama (local LLM - completely free)
- Groq (free tier - 30 requests/min)
- HuggingFace (free tier available)
"""

import json
import random
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class VideoScript:
    """Represents a complete video script."""
    title: str
    hook: str
    body: str
    call_to_action: str
    full_script: str
    keywords: List[str]
    hashtags: List[str]
    description: str
    estimated_duration: int  # in seconds
    niche: str
    

class ScriptPromptTemplates:
    """Templates for generating viral video scripts."""
    
    NICHES = {
        "motivation": {
            "topics": [
                "success mindset", "overcoming failure", "daily habits of winners",
                "mental toughness", "goal setting", "self-discipline",
                "morning routines", "productivity hacks", "confidence building",
                "dealing with rejection", "time management", "focus techniques"
            ],
            "hooks": [
                "The one thing successful people never tell you...",
                "I wasted 10 years before learning this...",
                "This simple habit changed my entire life...",
                "Stop doing this if you want to succeed...",
                "The harsh truth nobody wants to hear...",
                "Why 99% of people will never be successful...",
            ]
        },
        "facts": {
            "topics": [
                "psychology facts", "science discoveries", "history secrets",
                "animal facts", "space mysteries", "human body facts",
                "technology facts", "nature wonders", "mind-blowing statistics"
            ],
            "hooks": [
                "Scientists just discovered something terrifying...",
                "This fact will change how you see everything...",
                "99% of people don't know this about...",
                "The most disturbing fact about...",
                "What they don't teach you in school...",
                "This sounds fake but it's 100% real...",
            ]
        },
        "stories": {
            "topics": [
                "revenge stories", "karma stories", "wholesome moments",
                "plot twists", "life lessons", "inspiring journeys",
                "mysterious events", "unexpected outcomes", "true crime lite"
            ],
            "hooks": [
                "This story will give you chills...",
                "Nobody believed him until...",
                "She thought it was over, but then...",
                "The ending will shock you...",
                "This is the craziest thing I've ever heard...",
                "Wait for the plot twist...",
            ]
        },
        "tech": {
            "topics": [
                "AI developments", "hidden phone features", "tech tips",
                "future technology", "gadget reviews", "app recommendations",
                "cybersecurity tips", "coding facts", "tech history"
            ],
            "hooks": [
                "Your phone can do this and you had no idea...",
                "This AI tool is completely free and insane...",
                "Delete this app immediately if you have it...",
                "The future of technology is terrifying...",
                "This hidden feature will blow your mind...",
                "Stop using your phone wrong...",
            ]
        },
        "finance": {
            "topics": [
                "money saving tips", "investment basics", "passive income",
                "financial mistakes", "budgeting hacks", "side hustles",
                "credit score tips", "tax strategies", "wealth building"
            ],
            "hooks": [
                "Rich people do this and you don't...",
                "This money mistake is keeping you poor...",
                "I made $X doing this simple thing...",
                "Banks don't want you to know this...",
                "The easiest way to save money...",
                "Stop wasting money on this...",
            ]
        },
        "health": {
            "topics": [
                "sleep optimization", "nutrition facts", "exercise tips",
                "mental health", "longevity secrets", "body hacks",
                "stress management", "healthy habits", "wellness tips"
            ],
            "hooks": [
                "Doctors are finally admitting this...",
                "This one thing is destroying your health...",
                "Do this every morning for better health...",
                "The food industry doesn't want you to know...",
                "Why you're always tired (it's not sleep)...",
                "This simple change added years to my life...",
            ]
        },
        "mystery": {
            "topics": [
                "unsolved mysteries", "conspiracy theories", "paranormal",
                "unexplained events", "ancient mysteries", "disappearances",
                "strange phenomena", "creepy facts", "urban legends"
            ],
            "hooks": [
                "This mystery has never been solved...",
                "Scientists can't explain this...",
                "The truth about this will haunt you...",
                "Nobody talks about this anymore...",
                "This happened and was covered up...",
                "The creepiest thing ever recorded...",
            ]
        }
    }
    
    @classmethod
    def get_script_prompt(cls, niche: str, duration: int = 60) -> str:
        """Generate a prompt for creating a video script."""
        niche_data = cls.NICHES.get(niche, cls.NICHES["motivation"])
        topic = random.choice(niche_data["topics"])
        hook_style = random.choice(niche_data["hooks"])
        
        word_count = duration * 2.5  # Average speaking rate
        
        prompt = f"""Create a viral YouTube Shorts script about "{topic}" in the {niche} niche.

REQUIREMENTS:
1. Duration: Approximately {duration} seconds ({int(word_count)} words)
2. Format: YouTube Shorts (vertical video, fast-paced)
3. Hook Style: Similar to "{hook_style}"

STRUCTURE:
1. HOOK (first 3 seconds): An attention-grabbing opening that stops scrolling
2. BODY: Deliver valuable, engaging content with emotional peaks
3. CALL TO ACTION: End with engagement prompt (follow, like, comment)

STYLE GUIDELINES:
- Use short, punchy sentences
- Create curiosity gaps
- Include surprising facts or revelations
- Build tension and release
- Use power words that trigger emotions
- Make it feel personal and relatable
- Include a memorable takeaway

OUTPUT FORMAT (JSON):
{{
    "title": "Catchy title for the video (max 100 chars)",
    "hook": "The opening hook (first 3 seconds)",
    "body": "The main content",
    "call_to_action": "Engagement prompt at the end",
    "keywords": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"],
    "hashtags": ["#hashtag1", "#hashtag2", "#hashtag3"]
}}

Generate a unique, engaging script that would go viral:"""
        
        return prompt


class AIProvider(ABC):
    """Abstract base class for AI providers."""
    
    @abstractmethod
    async def generate(self, prompt: str) -> str:
        """Generate text from a prompt."""
        pass
    
    @abstractmethod
    def generate_sync(self, prompt: str) -> str:
        """Synchronous version of generate."""
        pass


class OllamaProvider(AIProvider):
    """Ollama local LLM provider - completely FREE."""
    
    def __init__(self, model: str = "llama3.2", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
    
    async def generate(self, prompt: str) -> str:
        """Generate text using Ollama API."""
        import aiohttp
        
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.8,
                "num_predict": 1000
            }
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("response", "")
                else:
                    raise Exception(f"Ollama API error: {response.status}")
    
    def generate_sync(self, prompt: str) -> str:
        """Synchronous generation using Ollama."""
        import requests
        
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.8,
                "num_predict": 1000
            }
        }
        
        response = requests.post(url, json=payload, timeout=120)
        if response.status_code == 200:
            data = response.json()
            return data.get("response", "")
        else:
            raise Exception(f"Ollama API error: {response.status_code}")


class GroqProvider(AIProvider):
    """Groq API provider - FREE tier (30 requests/min)."""
    
    def __init__(self, api_key: str, model: str = "llama-3.1-70b-versatile"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
    
    async def generate(self, prompt: str) -> str:
        """Generate text using Groq API."""
        import aiohttp
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.8,
            "max_tokens": 1000
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(self.base_url, headers=headers, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    return data["choices"][0]["message"]["content"]
                else:
                    raise Exception(f"Groq API error: {response.status}")
    
    def generate_sync(self, prompt: str) -> str:
        """Synchronous generation using Groq."""
        import requests
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.8,
            "max_tokens": 1000
        }
        
        response = requests.post(self.base_url, headers=headers, json=payload, timeout=60)
        if response.status_code == 200:
            data = response.json()
            return data["choices"][0]["message"]["content"]
        else:
            raise Exception(f"Groq API error: {response.status_code}")


class HuggingFaceProvider(AIProvider):
    """HuggingFace Inference API - FREE tier available."""
    
    def __init__(self, token: str, model: str = "mistralai/Mistral-7B-Instruct-v0.2"):
        self.token = token
        self.model = model
        self.base_url = f"https://api-inference.huggingface.co/models/{model}"
    
    async def generate(self, prompt: str) -> str:
        """Generate text using HuggingFace API."""
        import aiohttp
        
        headers = {"Authorization": f"Bearer {self.token}"}
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 1000,
                "temperature": 0.8,
                "return_full_text": False
            }
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(self.base_url, headers=headers, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    if isinstance(data, list) and len(data) > 0:
                        return data[0].get("generated_text", "")
                    return ""
                else:
                    raise Exception(f"HuggingFace API error: {response.status}")
    
    def generate_sync(self, prompt: str) -> str:
        """Synchronous generation using HuggingFace."""
        import requests
        
        headers = {"Authorization": f"Bearer {self.token}"}
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 1000,
                "temperature": 0.8,
                "return_full_text": False
            }
        }
        
        response = requests.post(self.base_url, headers=headers, json=payload, timeout=120)
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list) and len(data) > 0:
                return data[0].get("generated_text", "")
            return ""
        else:
            raise Exception(f"HuggingFace API error: {response.status_code}")


class ScriptGenerator:
    """Main script generator that uses AI to create viral video scripts."""
    
    def __init__(self, provider: AIProvider):
        self.provider = provider
    
    def _parse_script_response(self, response: str, niche: str, duration: int) -> VideoScript:
        """Parse the AI response into a VideoScript object."""
        # Try to extract JSON from the response
        json_match = re.search(r'\{[\s\S]*\}', response)
        
        if json_match:
            try:
                data = json.loads(json_match.group())
                
                # Construct full script
                full_script = f"{data.get('hook', '')} {data.get('body', '')} {data.get('call_to_action', '')}"
                
                # Generate description
                description = self._generate_description(
                    data.get('title', ''),
                    data.get('keywords', []),
                    data.get('hashtags', [])
                )
                
                return VideoScript(
                    title=data.get('title', 'Untitled Video'),
                    hook=data.get('hook', ''),
                    body=data.get('body', ''),
                    call_to_action=data.get('call_to_action', 'Follow for more!'),
                    full_script=full_script,
                    keywords=data.get('keywords', []),
                    hashtags=data.get('hashtags', ['#shorts', '#viral']),
                    description=description,
                    estimated_duration=duration,
                    niche=niche
                )
            except json.JSONDecodeError:
                logger.warning("Failed to parse JSON, using fallback parsing")
        
        # Fallback: parse as plain text
        return self._fallback_parse(response, niche, duration)
    
    def _fallback_parse(self, response: str, niche: str, duration: int) -> VideoScript:
        """Fallback parsing when JSON extraction fails."""
        lines = response.strip().split('\n')
        
        # Simple extraction
        title = lines[0] if lines else "Untitled Video"
        full_script = response
        
        return VideoScript(
            title=title[:100],
            hook=lines[0] if lines else "",
            body=" ".join(lines[1:-1]) if len(lines) > 2 else response,
            call_to_action=lines[-1] if lines else "Follow for more!",
            full_script=full_script,
            keywords=[niche, "viral", "trending"],
            hashtags=["#shorts", "#viral", f"#{niche}"],
            description=f"{title}\n\n#shorts #viral #{niche}",
            estimated_duration=duration,
            niche=niche
        )
    
    def _generate_description(self, title: str, keywords: List[str], hashtags: List[str]) -> str:
        """Generate a YouTube-optimized description."""
        hashtag_str = " ".join(hashtags[:10])
        keyword_str = ", ".join(keywords[:5])
        
        description = f"""{title}

{hashtag_str}

🔔 Turn on notifications to never miss a video!
👍 Like this video if you found it valuable
💬 Comment your thoughts below
📱 Share with someone who needs to see this

Keywords: {keyword_str}

#shorts #viral #trending #fyp"""
        
        return description
    
    def generate_script(self, niche: str, duration: int = 60) -> VideoScript:
        """Generate a video script synchronously."""
        prompt = ScriptPromptTemplates.get_script_prompt(niche, duration)
        
        logger.info(f"Generating script for niche: {niche}, duration: {duration}s")
        
        response = self.provider.generate_sync(prompt)
        script = self._parse_script_response(response, niche, duration)
        
        logger.info(f"Generated script: {script.title}")
        
        return script
    
    async def generate_script_async(self, niche: str, duration: int = 60) -> VideoScript:
        """Generate a video script asynchronously."""
        prompt = ScriptPromptTemplates.get_script_prompt(niche, duration)
        
        logger.info(f"Generating script for niche: {niche}, duration: {duration}s")
        
        response = await self.provider.generate(prompt)
        script = self._parse_script_response(response, niche, duration)
        
        logger.info(f"Generated script: {script.title}")
        
        return script
    
    def generate_batch(self, niche: str, count: int = 5, duration: int = 60) -> List[VideoScript]:
        """Generate multiple scripts at once."""
        scripts = []
        for i in range(count):
            try:
                script = self.generate_script(niche, duration)
                scripts.append(script)
                logger.info(f"Generated script {i+1}/{count}")
            except Exception as e:
                logger.error(f"Failed to generate script {i+1}: {e}")
        
        return scripts


def create_script_generator(config) -> ScriptGenerator:
    """Factory function to create a script generator based on config."""
    provider_name = config.ai.provider.lower()
    
    if provider_name == "ollama":
        provider = OllamaProvider(model=config.ai.model)
    elif provider_name == "groq":
        if not config.ai.groq_api_key:
            raise ValueError("Groq API key is required")
        provider = GroqProvider(api_key=config.ai.groq_api_key, model=config.ai.model)
    elif provider_name == "huggingface":
        if not config.ai.huggingface_token:
            raise ValueError("HuggingFace token is required")
        provider = HuggingFaceProvider(token=config.ai.huggingface_token)
    else:
        raise ValueError(f"Unknown AI provider: {provider_name}")
    
    return ScriptGenerator(provider)
