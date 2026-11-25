"""Execute only the eight reference functions, never arbitrary model-produced shell."""

import os
import shlex
import subprocess
from pathlib import Path

TOOLS = {
    "web_search": "Find a web page relevant to a query; return its URL.",
    "image_search": "Find an existing image relevant to a query; return its URL.",
    "image_gen": "Create a new image from a prompt; return its output path.",
    "ask_model": "Ask a second model for an independent opinion; return its answer.",
    "read_url": "Read the content at an HTTP or HTTPS URL; return a short excerpt.",
    "calc": "Evaluate numerical arithmetic with +, -, *, / and parentheses.",
    "remember": "Save a fact in this conversation's persistent memory file.",
    "recall": "Retrieve saved facts containing an exact substring; return Not found if absent.",
}
TOOLS_FILE = Path(__file__).with_name("tools.sh")


def command_parts(command: str) -> list[str]:
    """Split a command line into [name, argument]; raise ValueError for anything else."""
    parts = shlex.split(command)
    if len(parts) != 2 or parts[0] not in TOOLS:
        raise ValueError("Expected one known command with one argument")
    if not parts[1] or (parts[1].startswith("-") and parts[1] not in ("-h", "--help")):
        raise ValueError("Unsupported flag or empty argument")
    return parts


def command_name(command: str) -> str:
    """The first word of a command line, or INVALID when it does not parse."""
    try:
        return shlex.split(command)[0]
    except (ValueError, IndexError):
        return "INVALID"


def execute(command: str, memory: Path) -> str:
    """Requote parsed arguments; shell substitutions, redirects and operators stay literal."""
    parts = command_parts(command)
    result = subprocess.run(
        ["bash", "-c", f"source {shlex.quote(str(TOOLS_FILE))}; {shlex.join(parts)}"],
        env={**os.environ, "TOOLS_MEMORY": str(memory)},
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    if not result.stdout.strip():
        raise ValueError("Tool returned no result")
    return result.stdout.strip()
