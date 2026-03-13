import pyfiglet
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

console = Console()

def print_banner():
    ascii_banner = pyfiglet.figlet_format("D a t a g e n CLI")
    banner_text = Text(ascii_banner, style="bold magenta italic")

    panel = Panel(
        banner_text,
        border_style="bright_magenta",
        padding=(1, 4),
        title="[bold magenta]🚀 Welcome to[/bold magenta]",
        subtitle="[dim]CLI powered by Datagen[/dim]"
    )

    console.print(panel)

