import json
import os

PROGRESS_DIR = 'user_progress'


def save_progress(username: str, lesson_name: str, completed: bool):
    os.makedirs(PROGRESS_DIR, exist_ok=True)
    progress_path = os.path.join(PROGRESS_DIR, f'{username}.json')
    
    try:
        with open(progress_path, 'r') as f:
            progress = json.load(f)
    except FileNotFoundError:
        progress = {}

    progress[lesson_name] = completed
    
    with open(progress_path, 'w') as f:
        json.dump(progress, f)


def load_progress(username: str) -> dict:
    progress_path = os.path.join(PROGRESS_DIR, f'{username}.json')
    try:
        with open(progress_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}