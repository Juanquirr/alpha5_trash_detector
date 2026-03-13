import os
import json
import questionary
import requests
from rich.console import Console
from questionary import Style

console = Console()
CONFIG_PATH = "config.json"

SERVICES = [
    "api_key",
    "Datagen backend",
    "Diffusers service",
    "Segmentators service",
    "Autodistill service",
    "ComfyUI"
]

style = Style([
    ("qmark", "fg:#00ffff bold"),
    ("question", "fg:#ffffff bold"),
    ("answer", "fg:#00ff00 bold"),
    ("pointer", "fg:#ff00ff bold"),
    ("highlighted", "fg:#00ffff bold"),
    ("selected", "fg:#ffff00 bold"),
])

def check_url_accessible(url):
    try:
        requests.get(url, timeout=3)
        return True
    except Exception:
        return False

def prompt_for_valid_url(service_name):
    if service_name == "api_key":
        return questionary.password("🔑 Enter your API key:").ask()

    while True:
        url = questionary.text(f"🔗 Enter URL for {service_name}:", style=style).ask()

        if not url:
            console.print("[bold red]❌ URL cannot be empty. Try again.[/bold red]")
            continue

        console.print(f"[cyan]Checking {url}...[/cyan]")
        if check_url_accessible(url):
            console.print(f"✅ [green]{service_name} is reachable at {url}[/green]")
            return url
        else:
            console.print(f"❌ [red]{service_name} is NOT reachable. Try again.[/red]")

def create_config():
    config = {}
    console.print("\n🛠️ [bold cyan]Let's set up your service URLs[/bold cyan]\n")
    for service in SERVICES:
        config[service] = prompt_for_valid_url(service)

    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)

    console.print(f"\n💾 [cyan]Configuration saved to {CONFIG_PATH}[/cyan]")

def check_or_create_config():
    if not os.path.exists(CONFIG_PATH):
        console.print("[bold yellow]⚙️  Config file not found.[/bold yellow]")
        if questionary.confirm("Would you like to create a config file now?", default=True).ask():
            create_config()
        else:
            console.print("[red]❌ Cannot continue without config. Exiting.[/red]")
            exit(1)

def get_config_value(key: str) -> str:
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError("⚠️ config.json not found. Make sure to create it first.")

    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)

    if key not in config:
        raise KeyError(f"❌ Key '{key}' not found in config.json.")

    return config[key]
