import sys
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn
from rich import print as rprint
from orchestrator import LLMRouter

console = Console()

def main():
    console.print(Panel.fit(
        "[bold cyan]MELLM - LLM Router[/bold cyan]\n"
        "[italic white]Consumer-Hardware MoE Orchestration System (RTX 3050 Optimized)[/italic white]",
        border_style="bright_blue"
    ))
    
    try:
        router = LLMRouter()
    except Exception as e:
        console.print(f"[bold red]Initialization Error:[/bold red] {e}")
        sys.exit(1)
        
    console.print("[green]Router initialized and ready. Type 'exit' or 'quit' to stop.[/green]")
    
    while True:
        try:
            user_input = console.input("\n[bold yellow]Query:[/bold yellow] ")
            
            if user_input.lower() in ["exit", "quit"]:
                break
                
            if not user_input.strip():
                continue
                
            with console.status("[bold blue]Routing and generating response...") as status:
                result = router.query(user_input)
            
            if "error" in result:
                console.print(f"[bold red]Pipeline Error:[/bold red] {result['error']}")
                continue
                
            # Pretty-print results
            domain = result["domain"]
            confidence = result["confidence"]
            
            # Confidence bar
            conf_percent = int(confidence * 100)
            
            console.print(f"\n[bold cyan]Domain:[/bold cyan] {domain}")
            
            # Visual confidence bar
            with Progress(
                TextColumn("[bold white]Confidence:"),
                BarColumn(bar_width=30),
                TextColumn("{task.percentage:>3.0f}%"),
                console=console,
                transient=True
            ) as progress:
                task = progress.add_task("Conf", total=100)
                progress.update(task, completed=conf_percent)

            console.print(Panel(
                f"[bold white]{result['response']}[/bold white]",
                title="[bold green]Specialist Response[/bold green]",
                border_style="green"
            ))
            
            console.print(
                f"[dim white]Router rewrote as: {result['rewritten_prompt']}[/dim white]\n"
                f"[dim]Stats: Load Time: {result['load_time_seconds']}s | Inference Time: {result['inference_time_seconds']}s[/dim]"
            )
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {e}")

    console.print("\n[bold blue]Shutting down. Goodbye![/bold blue]")

if __name__ == "__main__":
    main()
