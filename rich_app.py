from transformers import pipeline
from rich import print
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, IntPrompt
from rich.spinner import Spinner
from concurrent.futures import ThreadPoolExecutor
from rich.live import Live
from rich.text import Text
import time
import logging

logging.basicConfig(level=logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

class UserWarningIgnoreFilter(logging.Filter):
    def filter(self, record):
        return "You have modified the pretrained model configuration to control generation." not in record.getMessage()

logging.getLogger("transformers").addFilter(UserWarningIgnoreFilter())


def clear_screen():
    console.clear()


def chat(model_name, input_text, max_length):
    model = model_pipelines[model_name]
    response = model(input_text, max_length=max_length)[0]["generated_text"]
    return response


def change_model_settings():
    global selected_model, max_length
    selected_model = Prompt.ask("Select a model", choices=model_names)
    max_length = IntPrompt.ask("Enter the max token length", default=1000)


def chat_with_spinner(model_name, input_text, max_length):
    with ThreadPoolExecutor() as executor:
        future = executor.submit(chat, model_name, input_text, max_length)
        
        with Live(console=console, auto_refresh=False) as live:
            spinner = "|/-\\"
            spin_idx = 0
            while not future.done():
                spin_idx = (spin_idx + 1) % len(spinner)
                live.update(Text(f"{spinner[spin_idx]} Waiting for model response...", style="bold cyan"))
                time.sleep(0.1)
                live.refresh()
    
    return future.result()




# Main loop for user interaction
model_pipelines = {
    "opt-6.7": pipeline("text-generation", model="distilgpt2"),
    "alpaca-7": pipeline("text-generation", model="distilgpt2"),
}

# Set up the Rich user interface
console = Console()
clear_screen()
panel_title = "Chat with AI Models"
model_names = list(model_pipelines.keys())
console.print(Panel.fit(panel_title, style="bold cyan"))
change_model_settings()

if __name__ == "__main__":

    while True:
        user_input = console.input("[bold green]You:[/bold green] ")

        if user_input.lower() == "exit":
            console.print("[bold red]Exiting...[/bold red]")
            break
        elif user_input.lower() == "settings":
            change_model_settings()
            clear_screen()
            console.print(Panel.fit(panel_title, style="bold cyan"))
            continue

        response = chat_with_spinner(selected_model, user_input, max_length)
        console.print(f"[bold magenta]{selected_model}:[/bold magenta] {response}\n")
