from datagen_sdk.client import DatagenClient
from .config import get_config_value
import questionary
from rich.console import Console

console = Console()

def ask_for_dataset_id() -> str:
    base_url = get_config_value("Datagen backend")
    api_key = get_config_value("api_key")  # Make sure this is also in your config!

    client = DatagenClient(api_key=api_key, base_url=base_url)

    while True:
        dataset_id = questionary.text("📦 Enter the Dataset ID:", qmark="🔍").ask()
        if not dataset_id or not dataset_id.strip():
            console.print("[bold red]❌ Dataset ID cannot be empty.[/bold red]")
            continue

        try:
            dataset = client.get_single_datasets(dataset_id.strip())
            console.print(f"[green]✅ Dataset found:[/] [bold]{dataset.name}[/bold]")
            return dataset_id.strip()
        except Exception as e:
            console.print(f"[bold red]❌ Could not find dataset with ID {dataset_id}[/bold red]")
            console.print(f"[dim]Details: {e}[/dim]")