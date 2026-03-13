import os

import questionary
from rich.console import Console

console = Console()

def ask_for_csv_file() -> str:
    while True:
        csv_path = questionary.path("📄 Enter path to CSV file:", qmark="🗂").ask()

        if not csv_path:
            console.print("[red]❌ You must enter a file path.[/red]")
            continue

        csv_path = os.path.expanduser(csv_path.strip())

        if not os.path.exists(csv_path):
            console.print(f"[red]❌ File not found at: {csv_path}[/red]")
        elif not csv_path.lower().endswith(".csv"):
            console.print(f"[yellow]⚠️ The file is not a .csv: {csv_path}[/yellow]")
        else:
            console.print(f"[green]✅ CSV file selected:[/] {csv_path}")
            return csv_path
