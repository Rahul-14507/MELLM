from dotenv import load_dotenv
load_dotenv()

import sys
import argparse
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.markup import escape
from rich.progress import Progress, BarColumn, TextColumn
from rich import print as rprint
from pathlib import Path
from setup.onboarding import run_onboarding, show_banner, VERSION
from orchestrator import LLMRouter

console = Console()

from rich.live import Live
from rich.text import Text
from rich.panel import Panel

def handle_query_streaming(router: LLMRouter, user_input: str, console: Console):
    """Handles a query with live streaming token output."""
    domain = "..."
    rewritten_prompt = ""
    load_time = 0.0
    inference_time = 0.0
    cache_hit = False
    context_turns = 0
    accumulated = ""
    done_event = {
        "response": "", "domain": domain,
        "inference_time_seconds": 0.0, "cache_hit": False,
        "context_turns": 0, "rewritten_prompt": "",
        "specialist_load_time": 0.0,
    }

    stream = router.stream_query(user_input)
    event = None  # Initialize to prevent UnboundLocalError

    try:
        # ── Phase 1: spin a status spinner during routing and model loading ─────────
        with console.status(
            "[cyan]Processing... (Router is persistent, specialist loads on-demand)",
            spinner="dots"
        ) as status:
            for event in stream:
                etype = event["type"]

                if etype == "routing":
                    domain = event.get("domain", domain)
                    rewritten_prompt = event.get("rewritten_prompt", "")
                    if event.get("is_multi_agent"):
                        status.update("[cyan]Multi-agent composition — routing to specialists...")
                    else:
                        status.update(
                            f"[cyan]Routed to [bold]{domain.upper()}[/bold] — loading specialist..."
                        )

                elif etype == "loaded":
                    load_time = event["load_time"]
                    cache_hit = event["cache_hit"]
                    cache_str = "HOT ♻" if cache_hit else f"loaded in {load_time:.2f}s"
                    status.update(f"[cyan]Specialist {cache_str} — generating response...")

                elif etype in ("token", "done"):
                    # Hand off to the Live streaming phase
                    break

        if event is None:
            console.print("[bold red]Error:[/bold red] Stream ended unexpectedly without yields.")
            return done_event

        # ── Phase 2: display tokens live (or full response for multi-agent) ─────────
        with Live(
            Panel("", title=f"[bold]Response (Specialist: {domain.lower()})[/bold]",
                  border_style="green"),
            console=console,
            refresh_per_second=15,
            vertical_overflow="visible"
        ) as live:
            # Handle the event that broke us out of Phase 1 first
            if event["type"] == "token":
                accumulated += event["content"]
                live.update(Panel(
                    accumulated,
                    title=f"[bold]Response (Specialist: {domain.lower()})[/bold]",
                    border_style="green"
                ))

            elif event["type"] == "done":
                done_event = event
                accumulated = event.get("response", "")
                live.update(Panel(
                    accumulated,
                    title=f"[bold]Response (Specialist: {domain.lower()})[/bold]",
                    border_style="green"
                ))

            # Continue consuming remaining stream events
            for event in stream:
                if event["type"] == "token":
                    accumulated += event["content"]
                    live.update(Panel(
                        accumulated,
                        title=f"[bold]Response (Specialist: {domain.lower()})[/bold]",
                        border_style="green"
                    ))
                elif event["type"] == "done":
                    done_event = event
                    inference_time = event.get("inference_time_seconds", 0.0)
                    cache_hit = event.get("cache_hit", False)
                    context_turns = event.get("context_turns", 0)
                    rewritten_prompt = event.get("rewritten_prompt", rewritten_prompt)
                    domain = event.get("domain", domain)
                    live.update(Panel(
                        event["response"],
                        title=f"[bold]Response (Specialist: {domain.lower()})[/bold]",
                        border_style="green"
                    ))
                    break
    except Exception as e:
        console.print(f"[bold red]Streaming Error:[/bold red] {e}")
        return done_event

    console.print(
        f"\n[dim]Router optimized prompt: {escape(rewritten_prompt[:80])}...[/dim]\n"
        f"[dim]Metrics: Router: resident (0s) | "
        f"Specialist Load: {load_time:.2f}s | "
        f"Inference: {inference_time:.2f}s | "
        f"Context: {context_turns} turns[/dim]"
    )
    return done_event

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
    router_filename = GGUF_REGISTRY.get(router_id, ("", router_id))[1]
    # Router is always loaded by the time this is called because it's persistent and init happens first
    status = "[bold green]Loaded (persistent)[/bold green]"
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
    # Show big ASCII banner first — always
    show_banner()
    
    # Check for --reconfigure flag
    if "--reconfigure" in sys.argv:
        run_onboarding(skip_banner=True)
        return
    
    # First run check — if no user_config.yaml, run onboarding
    if not Path("user_config.yaml").exists():
        console.print("[yellow]No user configuration found. Starting setup wizard...[/yellow]\n")
        run_onboarding(skip_banner=True)
    
    # Load config — prefer user_config.yaml over config.yaml
    config_path = "user_config.yaml" if Path("user_config.yaml").exists() else "config.yaml"
    
    parser = argparse.ArgumentParser(description="MELLM - LLM Router CLI")
    parser.add_argument("--preload", type=str, help="Preload a specific domain model or 'all'")
    parser.add_argument(
        "--web",
        action="store_true",
        help="Launch the web UI instead of the CLI"
    )
    args = parser.parse_args()

    try:
        # Initialize router — router model loads here and stays resident
        with console.status("[cyan]Loading router model (persistent)...[/cyan]"):
            router = LLMRouter(config_path=config_path)

        if args.web:
            import uvicorn
            from api import app, set_router
            set_router(router)  # inject the already-loaded router into the API
            console.print("\n[bold cyan]Web UI starting...[/bold cyan]")
            console.print("[dim]Open http://localhost:8000 in your browser[/dim]\n")
            import webbrowser, threading
            threading.Timer(1.5, lambda: webbrowser.open("http://localhost:8000")).start()
            uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
            return
            
        if args.preload:
            specialists = router.config["specialists"]
            target_domains = []
            
            if args.preload.lower() == "all":
                target_domains = list(specialists.keys())
            elif args.preload.lower() in specialists:
                target_domains = [args.preload.lower()]
            else:
                console.print(f"[bold red]Error:[/bold red] Invalid domain '{args.preload}'. Valid domains: {', '.join(specialists.keys())}, all")
                router.shutdown()
                sys.exit(1)
                
            for domain in target_domains:
                m_id = specialists[domain]["model_id"]
                console.print(f"[bold blue]Preloading {domain}...[/bold blue]")
                router.loader.get(m_id)
                router.loader.unload(m_id)
                console.print(f"[bold green]Done: {domain} cached.[/bold green]")
            
            console.print("\n[bold green]Preloading complete.[/bold green]")
            router.shutdown()
            sys.exit(0)

        # Show availability dashboard before starting normal mode
        show_availability(router)
        
    except Exception as e:
        console.print(f"[bold red]Initialization Error:[/bold red] {escape(str(e))}")
        sys.exit(1)
    console.print("[green]System initialized. Router is persistent. Type 'exit' or 'quit' to stop.[/green]")
    
    # Session stats tracked in CLI
    session_stats = {
        "total_queries": 0,
        "cache_hits": 0,
    }
    
    try:
        while True:
            try:
                user_input = console.input("\n[bold yellow]Query:[/bold yellow] ")
                
                if user_input.lower() in ["exit", "quit"]:
                    break
                    
                if user_input.lower() == "clear":
                    router.conversation_history.clear()
                    console.print("[yellow]Context cleared. Starting fresh conversation.[/yellow]")
                    continue

                if not user_input.strip():
                    continue
                    
                # Use the new streaming handler by default
                result = handle_query_streaming(router, user_input, console)
                
                if "error" in result:
                    console.print(f"[bold red]Pipeline Error:[/bold red] {result['error']}")
                    continue
                    
                domain = result["domain"]
                cache_hit = result.get("cache_hit", False)
                
                # Update session stats
                session_stats["total_queries"] += 1
                if cache_hit:
                    session_stats["cache_hits"] += 1

                # Calculate efficiency
                total = session_stats["total_queries"]
                hits = session_stats["cache_hits"]
                hit_rate = (hits / total * 100) if total > 0 else 0
                router_time_saved = (total - 1) * 1.0  # ~1s saved per query after first

                hot_label = "[green](HOT ♻)[/green]" if cache_hit else "[yellow](freshly loaded)[/yellow]"

                # ── Efficiency panel ──────────────────────────────────────────
                streak_display = " → ".join(router.domain_streak[-5:]) if router.domain_streak else "none"

                console.print(Panel(
                    f"[bold cyan]Session Efficiency[/bold cyan]\n"
                    f"  Queries this session : [white]{total}[/white]\n"
                    f"  Specialist cache hits: [green]{hits}/{total} ({hit_rate:.0f}%)[/green]\n"
                    f"  Router loads saved   : [green]{total - 1}[/green] "
                    f"(~[yellow]{router_time_saved:.1f}s[/yellow] saved)\n"
                    f"  Active specialist    : [cyan]{domain.upper()}[/cyan] {hot_label}\n"
                    f"  Context turns active : [cyan]{len(router.conversation_history)}/{router.max_history}[/cyan]\n"
                    f"  Domain streak        : [cyan]{streak_display}[/cyan]",
                    title=f"⚡ Efficiency ({VERSION})",
                    border_style="dim"
                ))
                
            except KeyboardInterrupt:
                console.print("\n[yellow]Interrupted.[/yellow]")
                break
            except Exception as e:
                console.print(f"[bold red]Error:[/bold red] {escape(str(e))}")
    finally:
        if 'router' in locals():
            router.shutdown()

if __name__ == "__main__":
    main()
