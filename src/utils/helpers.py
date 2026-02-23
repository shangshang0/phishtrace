"""
Utility functions for the phishing detection project
"""

import json
import os
from typing import Dict, List, Any


def load_json(filepath: str) -> Dict:
    """Load JSON file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Any, filepath: str, indent: int = 2):
    """Save data to JSON file"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def ensure_dir(directory: str):
    """Ensure directory exists"""
    os.makedirs(directory, exist_ok=True)


def count_files(directory: str, extension: str = None) -> int:
    """Count files in directory"""
    if not os.path.exists(directory):
        return 0
    
    files = os.listdir(directory)
    if extension:
        files = [f for f in files if f.endswith(extension)]
    return len(files)


def get_file_size(filepath: str) -> int:
    """Get file size in bytes"""
    if not os.path.exists(filepath):
        return 0
    return os.path.getsize(filepath)


def format_size(size_bytes: int) -> str:
    """Format bytes to human readable size"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"
