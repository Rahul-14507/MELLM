import os
import sys
import yaml
import torch
import psutil
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
from rich.prompt import Prompt, Confirm
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

MELLM_BANNER = """
███╗   ███╗███████╗██╗     ██╗     ███╗   ███╗
████╗ ████║██╔════╝██║     ██║     ████╗ ████║
██╔████╔██║█████╗  ██║     ██║     ██╔████╔██║
██║╚██╔╝██║██╔══╝  ██║     ██║     ██║╚██╔╝██║
██║ ╚═╝ ██║███████╗███████╗███████╗██║ ╚═╝ ██║
╚═╝     ╚═╝╚══════╝╚══════╝╚══════╝╚═╝     ╚═╝
"""

SUBTITLE = "Multi-Expert Large Language Model Router"
VERSION  = "v0.3.1"
TAGLINE  = "Consumer-Hardware MoE Orchestration · Run specialist LLMs locally"

DOMAIN_MODEL_OPTIONS = {
    "code": [
        {"label": "Fast",     "model_id": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
         "repo": "bartowski/Qwen2.5-Coder-1.5B-Instruct-GGUF",
         "file": "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf", "size": "1.5B"},
        {"label": "Balanced", "model_id": "Qwen/Qwen2.5-Coder-3B-Instruct",
         "repo": "bartowski/Qwen2.5-Coder-3B-Instruct-GGUF",
         "file": "Qwen2.5-Coder-3B-Instruct-Q4_K_M.gguf", "size": "3B"},
        {"label": "Strong",   "model_id": "Qwen/Qwen2.5-Coder-7B-Instruct",
         "repo": "bartowski/Qwen2.5-Coder-7B-Instruct-GGUF",
         "file": "Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf", "size": "7B"},
    ],
    "math": [
        {"label": "Fast",     "model_id": "Qwen/Qwen2.5-Math-1.5B-Instruct",
         "repo": "bartowski/Qwen2.5-Math-1.5B-Instruct-GGUF",
         "file": "Qwen2.5-Math-1.5B-Instruct-Q4_K_M.gguf", "size": "1.5B"},
        {"label": "Balanced", "model_id": "Qwen/Qwen2.5-Math-7B-Instruct",
         "repo": "bartowski/Qwen2.5-Math-7B-Instruct-GGUF",
         "file": "Qwen2.5-Math-7B-Instruct-Q4_K_M.gguf", "size": "7B"},
    ],
    "medical": [
        {"label": "Fast",     "model_id": "BioMistral/BioMistral-7B-DARE-GGUF",
         "repo": "BioMistral/BioMistral-7B-DARE-GGUF",
         "file": "ggml-model-Q2_K.gguf", "size": "7B Q2"},
        {"label": "Balanced", "model_id": "BioMistral/BioMistral-7B-DARE-GGUF",
         "repo": "BioMistral/BioMistral-7B-DARE-GGUF",
         "file": "ggml-model-Q4_K_M.gguf", "size": "7B Q4"},
    ],
    "legal": [
        {"label": "Fast",     "model_id": "AdaptLLM/law-LLM",
         "repo": "TheBloke/law-chat-GGUF",
         "file": "law-chat.Q2_K.gguf", "size": "7B Q2"},
        {"label": "Balanced", "model_id": "AdaptLLM/law-LLM",
         "repo": "mradermacher/magistrate-3.2-3b-it-GGUF",
         "file": "magistrate-3.2-3b-it.Q4_K_M.gguf", "size": "3B Q4"},
    ],
    "general": [
        {"label": "Fast",     "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
         "repo": "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
         "file": "qwen2.5-1.5b-instruct-q4_k_m.gguf", "size": "1.5B"},
        {"label": "Balanced", "model_id": "Qwen/Qwen2.5-3B-Instruct",
         "repo": "Qwen/Qwen2.5-3B-Instruct-GGUF",
         "file": "qwen2.5-3b-instruct-q4_k_m.gguf", "size": "3B"},
        {"label": "Strong",   "model_id": "Qwen/Qwen2.5-7B-Instruct",
         "repo": "bartowski/Qwen2.5-7B-Instruct-GGUF",
         "file": "Qwen2.5-7B-Instruct-Q4_K_M.gguf", "size": "7B"},
    ],
}

ROUTER_OPTIONS = [
    {
        "label": "Reliable",
        "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "repo": "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        "file": "qwen2.5-1.5b-instruct-q4_k_m.gguf",
        "size": "1.5B",
        "description": "Better JSON reliability, recommended for most users"
    },
    {
        "label": "Tiny",
        "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "repo": "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        "file": "qwen2.5-0.5b-instruct-q4_k_m.gguf",
        "size": "0.5B",
        "description": "Minimal VRAM (~400MB), occasional JSON failures"
    },
]

def get_hardware_info() -> dict:
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU"
    vram_gb = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1) if torch.cuda.is_available() else 0
    ram_gb = round(psutil.virtual_memory().total / 1e9, 1)
    
    if vram_gb < 4:
        rec = "1.5B only"
        tier = 0
    elif vram_gb < 6.5:
        rec = "1.5B – 3B"
        tier = 1
    elif vram_gb < 12.5:
        rec = "up to 7B"
        tier = 2
    else:
        rec = "up to 13B"
        tier = 3
        
    return {"gpu": gpu_name, "vram_gb": vram_gb, "ram_gb": ram_gb, "recommended": rec, "tier": tier}

def show_banner():
    console.print(f"[cyan]{MELLM_BANNER}[/cyan]")
    console.print(f"[bold white]{SUBTITLE}[/bold white] [dim white]({VERSION})[/dim white]")
    console.print(f"[dim white]{TAGLINE}[/dim white]")
    console.print("[cyan]" + "─" * 50 + "[/cyan]\n")

def _is_locally_cached(filename: str) -> bool:
    """Check if a GGUF file is already downloaded in the mellm cache."""
    cache_path = Path.home() / ".cache" / "mellm_gguf" / filename
    return cache_path.exists()

def print_model_options(options: list[dict], recommended_idx: int = 1) -> None:
    """Prints model options as a clean aligned rich table."""
    table = Table(
        box=box.SIMPLE,
        show_header=True,
        header_style="bold cyan",
        padding=(0, 2),
        show_edge=False
    )
    
    table.add_column("#", style="bold white", width=4, justify="right")
    table.add_column("Tier", width=10)
    table.add_column("Model", width=40)
    table.add_column("Size", width=8, justify="center")
    table.add_column("", width=14)  # status column: [LOCAL] / ← recommended
    
    for i, opt in enumerate(options, 1):
        is_local = _is_locally_cached(opt.get("file", ""))
        is_recommended = (i == recommended_idx + 1)
        
        tier_style = "green" if is_recommended else "white"
        local_badge = "[green][LOCAL][/green]" if is_local else ""
        rec_badge = "[cyan]← recommended[/cyan]" if is_recommended else ""
        status = f"{local_badge} {rec_badge}".strip()
        
        table.add_row(
            f"[{i}]",
            f"[{tier_style}]{opt['label']}[/{tier_style}]",
            opt["model_id"],
            opt["size"],
            status
        )
    
    # Always add Custom as the last option
    table.add_row(
        f"[{len(options) + 1}]",
        "Custom",
        "Enter your own HuggingFace model ID",
        "—",
        ""
    )
    
    console.print(table)

def print_domain_options() -> None:
    table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2), show_edge=False)
    table.add_column("#", style="bold white", width=4, justify="right")
    table.add_column("Domain", width=10, style="bold")
    table.add_column("Description", style="dim")
    
    domains = [
        ("Code",    "Programming, algorithms, debugging, data structures"),
        ("Math",    "Equations, calculus, proofs, complexity analysis"),
        ("Medical", "Health, symptoms, diagnosis, treatment"),
        ("Legal",   "Law, contracts, rights, lawsuits"),
        ("General", "Philosophy, history, concepts, explanations"),
    ]
    
    for i, (name, desc) in enumerate(domains, 1):
        table.add_row(f"[{i}]", name, desc)
    
    console.print(table)

def print_config_summary(config: dict) -> None:
    table = Table(
        box=box.ROUNDED,
        show_header=False,
        padding=(0, 2),
        title="[bold cyan]Your MELLM Configuration[/bold cyan]",
        border_style="cyan"
    )
    table.add_column("Component", style="bold white", width=12)
    table.add_column("Model", style="green")
    table.add_column("Size", style="dim", width=8)
    
    router_id = config.get("router", {}).get("model_id", "—")
    # Find size for router
    router_size = "—"
    for opt in ROUTER_OPTIONS:
        if opt["model_id"] == router_id:
            router_size = opt["size"]
            break
            
    table.add_row("Router", router_id, router_size)
    table.add_section()
    
    for domain, spec in config.get("specialists", {}).items():
        # Try to find size from defaults if possible
        model_id = spec.get("model_id", "—")
        size = spec.get("size", "—")
        if size == "—":
            for opt in DOMAIN_MODEL_OPTIONS.get(domain, []):
                if opt["model_id"] == model_id:
                    size = opt["size"]
                    break
        
        table.add_row(
            domain.capitalize(),
            model_id,
            size
        )
    
    console.print(table)

def run_onboarding(skip_banner=False):
    if not skip_banner:
        show_banner()
    console.print("[bold]Welcome to the MELLM Setup Wizard![/bold]")
    console.print("This will help you configure your local Mixture-of-Experts system.\n")
    
    # Step 1: hardware
    hw = get_hardware_info()
    console.print(Panel(
        f"  [bold]GPU[/bold]  : [cyan]{hw['gpu']}[/cyan]\n"
        f"  [bold]VRAM[/bold] : [cyan]{hw['vram_gb']} GB[/cyan]\n"
        f"  [bold]RAM[/bold]  : [cyan]{hw['ram_gb']} GB[/cyan]\n"
        f"  [bold]Recommended model size:[/bold] [yellow]{hw['recommended']}[/yellow]",
        title="Your Hardware",
        border_style="bright_blue",
        expand=False
    ))
    console.print("")
    
    # Step 2: Domains
    console.print("[bold cyan]Step 2 — Selecting Domains[/bold cyan]")
    print_domain_options()
    
    domains_all = ["code", "math", "medical", "legal", "general"]
    
    selection = Prompt.ask("\nSelect domains (e.g. 1,2,3 or Enter for all default)", default="all")
    if selection == "all":
        selected_domains = domains_all
    else:
        try:
            indices = [int(x.strip()) - 1 for x in selection.split(",")]
            selected_domains = [domains_all[i] for i in indices]
        except:
            console.print("[yellow]Invalid selection, defaulting to all.[/yellow]")
            selected_domains = domains_all
            
    # Step 3: Models per domain
    config = {
        "router": {"model_id": "", "max_new_tokens": 256},
        "specialists": {},
        "gguf_registry": {}
    }
    
    for domain in selected_domains:
        console.print(f"\n[bold cyan]Step 3 — Configure {domain.upper()} Specialist[/bold cyan]")
        options = DOMAIN_MODEL_OPTIONS.get(domain, [])
        rec_idx = min(hw['tier'], len(options) - 1)
        
        print_model_options(options, recommended_idx=rec_idx)
        
        choice = Prompt.ask(f"\nSelect [1-{len(options)+1}] or press Enter for recommended", default=str(rec_idx + 1))
        
        if choice == str(len(options)+1):
            repo = Prompt.ask("  Enter HuggingFace repo ID (e.g. bartowski/MyModel-GGUF)")
            file = Prompt.ask("  Enter GGUF filename (e.g. mymodel-Q4_K_M.gguf)")
            model_id = Prompt.ask("  Enter a unique model ID for this", default=repo.split('/')[-1])
            config["specialists"][domain] = {"model_id": model_id, "max_new_tokens": 2048, "size": "Custom"}
            config["gguf_registry"][model_id] = [repo, file]
        else:
            idx = int(choice) - 1
            opt = options[idx]
            config["specialists"][domain] = {"model_id": opt["model_id"], "max_new_tokens": 2048, "size": opt["size"]}
            config["gguf_registry"][opt["model_id"]] = [opt["repo"], opt["file"]]

    console.print("\n[bold cyan]Step 4 — Configure Router[/bold cyan]")
    console.print("The router classifies your queries and decides which specialist to load.")
    # For router, recommendation is always 1 (Reliable)
    print_model_options(ROUTER_OPTIONS, recommended_idx=0)
    
    choice = Prompt.ask("\nSelect [1-2] or press Enter for recommended", default="1")
    if choice == "3": # Custom router? Why not.
         repo = Prompt.ask("  Enter HuggingFace repo ID")
         file = Prompt.ask("  Enter GGUF filename")
         model_id = Prompt.ask("  Enter model ID", default=repo.split('/')[-1])
         config["router"]["model_id"] = model_id
         config["gguf_registry"][model_id] = [repo, file]
    else:
        opt = ROUTER_OPTIONS[int(choice)-1]
        config["router"]["model_id"] = opt["model_id"]
        config["router"]["size"] = opt["size"]
        config["gguf_registry"][opt["model_id"]] = [opt["repo"], opt["file"]]
    
    # Step 5: HF Token
    console.print("\n[bold cyan]Step 5 — HuggingFace Token[/bold cyan]")
    console.print("Optional but recommended for faster downloads and private models.")
    console.print("Get yours at: https://huggingface.co/settings/tokens")
    token = Prompt.ask("Enter token (or press Enter to skip)", password=True, default="")
    
    if token:
        with open(".env", "a") as f:
            f.write(f"\nHF_TOKEN={token}\n")
        console.print("[green]Token saved to .env[/green]")

    # Step 6: Summary
    console.print("\n[bold cyan]Step 6 — Configuration Summary[/bold cyan]")
    print_config_summary(config)
    
    if Confirm.ask("\nSave this configuration?", default=True):
        with open("user_config.yaml", "w") as f:
            # Remove helper 'size' field from specialist entries before saving
            save_config = {
                "router": {k: v for k, v in config["router"].items() if k != "size"},
                "specialists": {d: {k: v for k, v in s.items() if k != "size"} for d, s in config["specialists"].items()},
                "gguf_registry": config["gguf_registry"]
            }
            yaml.dump(save_config, f, sort_keys=False)
        console.print("\n[bold green]Configuration saved to user_config.yaml![/bold green]")
        console.print("You can now run MELLM using: [cyan]python cli.py[/cyan]")
    else:
        console.print("[yellow]Setup cancelled. No changes were saved.[/yellow]")

if __name__ == "__main__":
    run_onboarding()
