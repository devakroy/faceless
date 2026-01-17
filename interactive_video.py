#!/usr/bin/env python3
"""
Interactive Faceless YouTube Video Creator
Asks for video type and duration, then creates the video
"""

import time
from src.pipeline.automation import FacelessAutomation
from src.utils.config_loader import get_config
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

def main():
    console.print("\n[bold blue]🎬 Interactive Faceless Video Creator[/bold blue]\n")
    
    # Ask for video type
    console.print("[bold]Step 1: Choose Video Type[/bold]")
    video_types = {
        '1': ('motivation', 'Motivation/Inspiration'),
        '2': ('facts', 'Interesting Facts'),
        '3': ('stories', 'Engaging Stories'),
        '4': ('tech', 'Technology Tips'),
        '5': ('other', 'Other (custom)')
    }
    
    for key, (value, desc) in video_types.items():
        console.print(f"  {key}. {desc}")
    
    choice = input("\nEnter your choice (1-5): ")
    
    if choice == '5':
        niche = input("Enter custom niche name: ")
    else:
        niche = video_types.get(choice, ('motivation', ''))[0]
    
    # Ask for duration
    console.print("\n[bold]Step 2: Choose Duration[/bold]")
    duration = int(input("Enter duration in seconds (15-60 recommended): ") or 20)
    
    # Update config
    config = get_config()
    config.channel.niche = niche
    config.channel.target_duration = duration
    
    # Create automation
    automation = FacelessAutomation()
    
    # Create video with progress
    console.print(f"\n[bold]Creating {duration}-second {niche} video...[/bold]")
    console.print("This may take a few minutes. Please wait...\n")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Generating script...", total=None)
        
        try:
            result = automation.create_video(niche=niche)
            
            if result.success:
                console.print(f"\n[bold green]✅ Video Created Successfully![/bold green]")
                console.print(f"   Title: {result.title}")
                console.print(f"   Path: {result.video_path}")
                console.print(f"   Duration: {result.duration:.2f}s")
                console.print(f"\n   🎥 Video saved to: output/{result.video_path.split('/')[-1]}")
            else:
                console.print(f"\n[bold red]❌ Video Creation Failed[/bold red]")
                console.print(f"   Error: {result.error}")
                
        except Exception as e:
            console.print(f"\n[bold red]❌ Error: {e}[/bold red]")

if __name__ == "__main__":
    main()
