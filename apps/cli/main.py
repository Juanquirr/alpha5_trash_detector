from apps.cli.dataset import ask_for_dataset_id
from apps.cli.dataset_definition import ask_for_csv_file
from apps.cli.pipeline1 import execute_pipeline1
from apps.cli.pipeline3 import execute_pipeline3
from .banner import print_banner
from .config import check_or_create_config
from .pipeline import select_pipeline
from rich.console import Console
import pandas as pd
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn


console = Console()


def main():
    print_banner()
    check_or_create_config()

    dataset_id = ask_for_dataset_id()
    choice = select_pipeline()
    if choice:
        console.print(f"\n✅ [bold green]You selected:[/] {choice}")
        csv_path = ask_for_csv_file()

        try:
            df = pd.read_csv(csv_path)
            console.print(f"[bold cyan]📄 Loaded CSV with {len(df)} rows.[/bold cyan]\n")

            for index, row in df.iterrows():
                data = row.to_dict()
                console.print(f"[blue]▶ Generating image {index + 1}/{len(df)}[/blue]")
                if(choice =="Pipeline 1"): execute_pipeline1(data, dataset_id=dataset_id)
                if(choice =="Pipeline 3"): execute_pipeline3(data, dataset_id=dataset_id)
        except Exception as e:
            console.print(f"[bold red]❌ Failed to generate data:[/bold red] {e}")
    else:
        console.print("\n❌ [bold red]No option selected. Exiting.[/]")


if __name__ == "__main__":
    main()
