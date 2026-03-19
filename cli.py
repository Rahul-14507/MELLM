from dotenv import load_dotenv
load_dotenv()

from dotenv import load_dotenv
load_dotenv()

import sys
import argparse
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn
from rich import print as rprint
from orchestrator import LLMRouter

console = Console()

def show_availability(router: LLMRouter):
    """Displays a dashboard of which models are already downloaded."""
    from loader.airllm_loader import GGUF_REGISTRY
    availability = router.loader.get_local_models()
    
    table = Table(title="Model Availability Dashboard", border_style="cyan")
    table.add_column("Type", style="bold")
    table.add_column("Domain", style="magenta")
    table.add_column("Model (GGUF)", style="white")
    table.add_column("Status", justify="center")

    # Router
    router_id = router.config["router"]["model_id"]
    is_router_local = availability.get(router_id)
    router_filename = GGUF_REGISTRY.get(router_id, ("", router_id))[1]
    status = "[green]Ready[/green]" if is_router_local else "[yellow]Download Required[/yellow]"
    table.add_row("Router", "N/A", router_filename, status)
    
    # Specialists
    for domain, spec in router.config["specialists"].items():
        m_id = spec["model_id"]
        is_local = availability.get(m_id)
        filename = GGUF_REGISTRY.get(m_id, ("", m_id))[1]
        status = "[green]Ready[/green]" if is_local else "[yellow]Download Required[/yellow]"
        table.add_row("Specialist", domain.capitalize(), filename, status)
        
    console.print(table)
    console.print("[dim italic white]* Download will happen automatically on first query for each domain.[/dim italic white]\n")

def main():
    parser = argparse.ArgumentParser(description="MELLM - LLM Router CLI")
    parser.add_argument("--preload", type=str, help="Preload a specific domain model or 'all'")
    args = parser.parse_args()

    console.print(Panel.fit(
        "[bold cyan]MELLM - LLM Router[/bold cyan]\n"
        "[italic white]Consumer-Hardware MoE Orchestration System[/italic white]",
        border_style="bright_blue"
    ))
    
    try:
        # Initialize router (orchestrator handles config loading)
        router = LLMRouter()
        
        if args.preload:
            specialists = router.config["specialists"]
            target_domains = []
            
            if args.preload.lower() == "all":
                target_domains = list(specialists.keys())
            elif args.preload.lower() in specialists:
                target_domains = [args.preload.lower()]
            else:
                console.print(f"[bold red]Error:[/bold red] Invalid domain '{args.preload}'. Valid domains: {', '.join(specialists.keys())}, all")
                sys.exit(1)
                
            for domain in target_domains:
                m_id = specialists[domain]["model_id"]
                console.print(f"[bold blue]Preloading {domain}...[/bold blue]")
                # Use standard get/unload cycle to trigger download and slicing
                router.loader.get(m_id)
                router.loader.unload(m_id)
                console.print(f"[bold green]Done: {domain} cached.[/bold green]")
            
            console.print("\n[bold green]Preloading complete.[/bold green]")
            sys.exit(0)

        # Show availability dashboard before starting normal mode
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

            from rich.markup import escape
            
            console.print(Panel(
                Text(result['response'], style="bold white"),
                title=f"[bold green]Response (Specialist: {domain})[/bold green]",
                border_style="green"
            ))
            
            console.print(
                f"[dim white]Router optimized prompt: {escape(result['rewritten_prompt'][:100])}...[/dim white]\n"
                f"[dim]Metrics: Router Load: {result['router_load_time']}s | Specialist Load: {result['specialist_load_time']}s | Inference: {result['inference_time_seconds']}s[/dim]"
            )
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            console.print(f"[bold red]Error durante la ejecución:[/bold red] {escape(str(e))}")

    console.print("\n[bold blue]VRAM cleared. Goodbye![/bold blue]")

if __name__ == "__main__":
    main()
