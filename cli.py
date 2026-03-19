import sys
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn
from rich import print as rprint
from orchestrator import LLMRouter

console = Console()

def show_availability(router: LLMRouter):
    """Displays a dashboard of which models are already downloaded."""
    availability = router.loader.get_local_models()
    
    table = Table(title="Model Availability Dashboard", border_style="cyan")
    table.add_column("Type", style="bold")
    table.add_column("Domain", style="magenta")
    table.add_column("Model ID", style="white")
    table.add_column("Status", justify="center")

    # Router
    router_id = router.config["router"]["model_id"]
    status = "[green]Ready[/green]" if availability.get(router_id) else "[yellow]Download Required[/yellow]"
    table.add_row("Router", "N/A", router_id, status)
    
    # Specialists
    for domain, spec in router.config["specialists"].items():
        m_id = spec["model_id"]
        status = "[green]Ready[/green]" if availability.get(m_id) else "[yellow]Download Required[/yellow]"
        table.add_row("Specialist", domain.capitalize(), m_id, status)
        
    console.print(table)
    console.print("[dim italic white]* Download will happen automatically on first query for each domain.[/dim italic white]\n")

def main():
    console.print(Panel.fit(
        "[bold cyan]MELLM - LLM Router[/bold cyan]\n"
        "[italic white]Consumer-Hardware MoE Orchestration System[/italic white]",
        border_style="bright_blue"
    ))
    
    try:
        # Initialize router (orchestrator handles config loading)
        router = LLMRouter()
        
        # Show availability dashboard before starting
        show_availability(router)
        
    except Exception as e:
        console.print(f"[bold red]Initialization Error:[/bold red] {e}")
        sys.exit(1)
        
    console.print("[green]System initialized. Type 'exit' or 'quit' to stop.[/green]")
    
    while True:
        try:
            user_input = console.input("\n[bold yellow]Query:[/bold yellow] ")
            
            if user_input.lower() in ["exit", "quit"]:
                break
                
            if not user_input.strip():
                continue
                
            # Note: Progress bar for downloads are handled by HuggingFace Hub internally
            # We use a status message for the orchestration phases
            with console.status("[bold blue]Processing... (Both models will load/unload on-demand)") as status:
                result = router.query(user_input)
            
            if "error" in result:
                console.print(f"[bold red]Pipeline Error:[/bold red] {result['error']}")
                continue
                
            domain = result["domain"]
            confidence = result["confidence"]
            conf_percent = int(confidence * 100)
            
            console.print(f"\n[bold cyan]Domain:[/bold cyan] {domain.upper()}")
            
            # Confidence bar
            with Progress(
                TextColumn("[bold white]Confidence:"),
                BarColumn(bar_width=30, complete_style="cyan", finished_style="bright_cyan"),
                TextColumn("{task.percentage:>3.0f}%"),
                console=console,
                transient=True
            ) as progress:
                task = progress.add_task("Conf", total=100)
                progress.update(task, completed=conf_percent)

            console.print(Panel(
                f"[bold white]{result['response']}[/bold white]",
                title=f"[bold green]Response (Specialist: {domain})[/bold green]",
                border_style="green"
            ))
            
            console.print(
                f"[dim white]Router optimized prompt: {result['rewritten_prompt'][:100]}...[/dim white]\n"
                f"[dim]Metrics: Router Load: {result['router_load_time']}s | Specialist Load: {result['specialist_load_time']}s | Inference: {result['inference_time_seconds']}s[/dim]"
            )
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            console.print(f"[bold red]Error durante la ejecución:[/bold red] {e}")

    console.print("\n[bold blue]VRAM cleared. Goodbye![/bold blue]")

if __name__ == "__main__":
    main()
