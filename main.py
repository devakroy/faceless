#!/usr/bin/env python3
"""
Faceless YouTube Channel Automation
====================================
A fully automated, FREE solution for creating and managing
a faceless YouTube channel that can grow organically.

Usage:
    python main.py create          # Create a single video
    python main.py batch 5         # Create 5 videos
    python main.py start           # Start 24/7 automation
    python main.py status          # Check automation status
    python main.py insights        # Get growth insights
    python main.py setup           # Interactive setup wizard

All processing runs locally on your CPU/GPU.
All APIs used are FREE.
"""

import sys
import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import print as rprint

console = Console()


@click.group()
@click.version_option(version="1.0.0")
def cli():
    """🎬 Faceless YouTube Channel Automation - 100% FREE"""
    pass


@cli.command()
@click.option('--niche', '-n', default=None, help='Content niche (motivation, facts, stories, etc.)')
def create(niche):
    """Create a single video and upload to YouTube."""
    from src.pipeline.automation import create_automation
    
    console.print("\n[bold blue]🎬 Creating Video...[/bold blue]\n")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Initializing...", total=None)
        
        try:
            automation = create_automation()
            
            progress.update(task, description="Generating script...")
            result = automation.create_video(niche)
            
            if result.success:
                console.print(f"\n[bold green]✅ Video Created Successfully![/bold green]")
                console.print(f"   Title: {result.title}")
                console.print(f"   Video ID: {result.video_id}")
                console.print(f"   Path: {result.video_path}")
                console.print(f"   Duration: {result.duration:.2f}s")
            else:
                console.print(f"\n[bold red]❌ Video Creation Failed[/bold red]")
                console.print(f"   Error: {result.error}")
                
        except Exception as e:
            console.print(f"\n[bold red]❌ Error: {e}[/bold red]")
            sys.exit(1)


@cli.command()
@click.argument('count', type=int, default=3)
@click.option('--niches', '-n', multiple=True, help='Niches to cycle through')
def batch(count, niches):
    """Create multiple videos in batch."""
    from src.pipeline.automation import create_automation
    
    console.print(f"\n[bold blue]🎬 Creating {count} Videos...[/bold blue]\n")
    
    try:
        automation = create_automation()
        niches_list = list(niches) if niches else None
        
        results = automation.create_batch(count, niches_list)
        
        # Summary table
        table = Table(title="Batch Results")
        table.add_column("#", style="cyan")
        table.add_column("Title", style="white")
        table.add_column("Status", style="green")
        table.add_column("Duration", style="yellow")
        
        success_count = 0
        for i, result in enumerate(results, 1):
            status = "✅ Success" if result.success else f"❌ {result.error[:30]}"
            title = result.title[:40] + "..." if result.title and len(result.title) > 40 else (result.title or "N/A")
            table.add_row(
                str(i),
                title,
                status,
                f"{result.duration:.1f}s"
            )
            if result.success:
                success_count += 1
        
        console.print(table)
        console.print(f"\n[bold]Summary: {success_count}/{count} videos created successfully[/bold]")
        
    except Exception as e:
        console.print(f"\n[bold red]❌ Error: {e}[/bold red]")
        sys.exit(1)


@cli.command()
def start():
    """Start 24/7 automated video creation."""
    from src.pipeline.automation import create_automation
    
    console.print(Panel.fit(
        "[bold green]🚀 Starting Faceless Automation[/bold green]\n\n"
        "The system will automatically:\n"
        "• Generate viral scripts using AI\n"
        "• Create videos with TTS and subtitles\n"
        "• Upload to YouTube at optimal times\n"
        "• Track analytics and optimize\n\n"
        "[yellow]Press Ctrl+C to stop[/yellow]",
        title="Faceless YouTube Automation"
    ))
    
    try:
        automation = create_automation()
        automation.start_automation()
        
        console.print("\n[green]✅ Automation is running![/green]")
        console.print("Check logs/automation.log for details\n")
        
        # Keep running
        import time
        while True:
            time.sleep(60)
            status = automation.get_status()
            console.print(f"[dim]Status: {status['scheduler']['videos_today']} videos today, "
                         f"next run: {status['scheduler']['next_scheduled']}[/dim]")
            
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopping automation...[/yellow]")
        automation.stop_automation()
        console.print("[green]Automation stopped gracefully[/green]")
    except Exception as e:
        console.print(f"\n[bold red]❌ Error: {e}[/bold red]")
        sys.exit(1)


@cli.command()
def status():
    """Check current automation status."""
    from src.pipeline.automation import create_automation
    
    try:
        automation = create_automation()
        status = automation.get_status()
        
        # Scheduler status
        sched = status['scheduler']
        console.print(Panel.fit(
            f"[bold]Running:[/bold] {'✅ Yes' if sched['is_running'] else '❌ No'}\n"
            f"[bold]Videos Today:[/bold] {sched['videos_today']}\n"
            f"[bold]Total Videos:[/bold] {sched['total_videos']}\n"
            f"[bold]Last Run:[/bold] {sched['last_run'] or 'Never'}\n"
            f"[bold]Next Scheduled:[/bold] {sched['next_scheduled'] or 'Not scheduled'}",
            title="📊 Automation Status"
        ))
        
        # Channel info
        channel = status['channel']
        console.print(Panel.fit(
            f"[bold]Name:[/bold] {channel['name']}\n"
            f"[bold]Niche:[/bold] {channel['niche']}\n"
            f"[bold]Format:[/bold] {channel['format']}",
            title="📺 Channel Info"
        ))
        
    except Exception as e:
        console.print(f"\n[bold red]❌ Error: {e}[/bold red]")


@cli.command()
def insights():
    """Get growth insights and recommendations."""
    from src.pipeline.automation import create_automation
    
    try:
        automation = create_automation()
        insights = automation.get_insights()
        
        if not insights:
            console.print("[yellow]No insights available yet. Create more videos first![/yellow]")
            return
        
        console.print("\n[bold blue]📈 Growth Insights[/bold blue]\n")
        
        for insight in insights:
            priority_color = {
                'high': 'red',
                'medium': 'yellow',
                'low': 'green'
            }.get(insight['priority'], 'white')
            
            console.print(Panel(
                f"[bold]{insight['insight']}[/bold]\n\n"
                f"[italic]Action: {insight['action']}[/italic]",
                title=f"[{priority_color}]{insight['category'].upper()}[/{priority_color}]",
                border_style=priority_color
            ))
        
    except Exception as e:
        console.print(f"\n[bold red]❌ Error: {e}[/bold red]")


@cli.command()
def setup():
    """Interactive setup wizard."""
    console.print(Panel.fit(
        "[bold green]🔧 Faceless YouTube Setup Wizard[/bold green]\n\n"
        "This wizard will help you configure your faceless channel.",
        title="Setup"
    ))
    
    # Check dependencies
    console.print("\n[bold]Checking dependencies...[/bold]")
    
    dependencies = [
        ("FFmpeg", "ffmpeg -version"),
        ("Python packages", "pip show moviepy"),
    ]
    
    import subprocess
    all_ok = True
    
    for name, cmd in dependencies:
        try:
            subprocess.run(cmd.split(), capture_output=True, check=True)
            console.print(f"  ✅ {name}")
        except Exception:
            console.print(f"  ❌ {name} - [red]Not installed[/red]")
            all_ok = False
    
    if not all_ok:
        console.print("\n[yellow]Please install missing dependencies first.[/yellow]")
        console.print("Run: pip install -r requirements.txt")
        console.print("And install FFmpeg for your system.")
        return
    
    # Configuration
    console.print("\n[bold]Configuration:[/bold]")
    
    from pathlib import Path
    config_path = Path("config/config.yaml")
    
    if config_path.exists():
        console.print(f"  ✅ Config file exists: {config_path}")
    else:
        console.print(f"  ❌ Config file missing: {config_path}")
        console.print("  Creating default config...")
    
    # API Keys check
    console.print("\n[bold]API Keys (Optional but recommended):[/bold]")
    console.print("  • Pexels API: https://www.pexels.com/api/ (FREE)")
    console.print("  • Pixabay API: https://pixabay.com/api/docs/ (FREE)")
    console.print("  • Groq API: https://console.groq.com/ (FREE tier)")
    
    # YouTube setup
    console.print("\n[bold]YouTube Setup:[/bold]")
    console.print("  1. Go to https://console.cloud.google.com/")
    console.print("  2. Create a new project")
    console.print("  3. Enable YouTube Data API v3")
    console.print("  4. Create OAuth 2.0 credentials")
    console.print("  5. Download and save as config/client_secrets.json")
    
    # Ollama setup
    console.print("\n[bold]AI Setup (Ollama - FREE local LLM):[/bold]")
    console.print("  1. Install Ollama: curl -fsSL https://ollama.com/install.sh | sh")
    console.print("  2. Pull a model: ollama pull llama3.2")
    console.print("  3. Start Ollama: ollama serve")
    
    console.print("\n[green]Setup complete! Edit config/config.yaml to customize.[/green]")
    console.print("Then run: python main.py create")


@cli.command()
def test():
    """Test all components without uploading."""
    console.print("\n[bold blue]🧪 Testing Components...[/bold blue]\n")
    
    tests = []
    
    # Test config
    try:
        from src.utils.config_loader import get_config
        config = get_config()
        tests.append(("Config Loading", True, ""))
    except Exception as e:
        tests.append(("Config Loading", False, str(e)))
    
    # Test AI
    try:
        from src.ai.script_generator import OllamaProvider
        provider = OllamaProvider()
        # Just check if Ollama is reachable
        import requests
        requests.get("http://localhost:11434/api/tags", timeout=5)
        tests.append(("Ollama AI", True, ""))
    except Exception as e:
        tests.append(("Ollama AI", False, "Ollama not running. Start with: ollama serve"))
    
    # Test TTS
    try:
        import subprocess
        result = subprocess.run(['edge-tts', '--version'], capture_output=True)
        tests.append(("Edge TTS", True, ""))
    except Exception:
        tests.append(("Edge TTS", False, "Install with: pip install edge-tts"))
    
    # Test FFmpeg
    try:
        import subprocess
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        tests.append(("FFmpeg", True, ""))
    except Exception:
        tests.append(("FFmpeg", False, "Install FFmpeg for your system"))
    
    # Test MoviePy
    try:
        from moviepy.editor import ColorClip
        tests.append(("MoviePy", True, ""))
    except Exception as e:
        tests.append(("MoviePy", False, str(e)))
    
    # Display results
    table = Table(title="Component Tests")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="white")
    table.add_column("Notes", style="yellow")
    
    for name, success, note in tests:
        status = "[green]✅ Pass[/green]" if success else "[red]❌ Fail[/red]"
        table.add_row(name, status, note[:50])
    
    console.print(table)
    
    passed = sum(1 for _, s, _ in tests if s)
    console.print(f"\n[bold]Results: {passed}/{len(tests)} tests passed[/bold]")


if __name__ == "__main__":
    cli()
