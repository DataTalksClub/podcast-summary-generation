import os
from pathlib import Path
from typing import Union


def load_text(path: Union[str, Path]) -> str:
    """Load text content from a file with UTF-8 encoding."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {path}")
    except Exception as e:
        raise RuntimeError(f"Failed to load text from {path}: {e}")

def save_text(content: str, path: Union[str, Path], create_dirs: bool = True) -> None:
    """Save text content to a file with UTF-8 encoding."""
    try:
        path = Path(path)
        if create_dirs:
            os.makedirs(path.parent, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
    except Exception as e:
        raise RuntimeError(f"Failed to save text to {path}: {e}")