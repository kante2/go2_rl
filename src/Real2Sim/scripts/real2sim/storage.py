"""Publish completed result directories, restoring the previous result on failure."""
from pathlib import Path
import os
import shutil
import tempfile


def publish_directory(stage, destination):
    stage, destination = Path(stage), Path(destination)
    backup = None
    if destination.exists():
        backup = Path(tempfile.mkdtemp(prefix=f'.{destination.name}-previous-', dir=destination.parent))
        backup.rmdir()
        os.replace(destination, backup)
    try:
        os.replace(stage, destination)
    except BaseException:
        if backup is not None:
            os.replace(backup, destination)
        raise
    if backup is not None:
        shutil.rmtree(backup)
