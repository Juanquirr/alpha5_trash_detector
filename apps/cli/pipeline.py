import questionary
from questionary import Style

style = Style([
    ("qmark", "fg:#00ffff bold"),
    ("question", "fg:#ffffff bold"),
    ("answer", "fg:#00ff00 bold"),
    ("pointer", "fg:#ff00ff bold"),
    ("highlighted", "fg:#00ffff bold"),
    ("selected", "fg:#ffff00 bold"),
])

def select_pipeline():
    return questionary.select(
        "💡 Please select a pipeline to run:",
        choices=[
            "Pipeline 1",
            "Pipeline 2",
            "Pipeline 3"
        ],
        style=style
    ).ask()
