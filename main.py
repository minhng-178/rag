import os
import sys
from pathlib import Path

# Automatically switch to virtual environment if it exists and we're not inside it
root_dir = Path(__file__).resolve().parent
venv_dir = root_dir / "venv"
venv_python = venv_dir / "bin" / "python"
if sys.platform == "win32":
    venv_python = venv_dir / "Scripts" / "python.exe"

if venv_python.exists():
    if Path(sys.prefix).resolve() != venv_dir.resolve():
        print(f"Re-executing with virtual environment python: {venv_python}")
        os.execv(str(venv_python), [str(venv_python)] + sys.argv)

# Add 'src' to the Python path to ensure 'rag' can be imported
# even if it wasn't installed in editable mode.
src_dir = root_dir / "src"
if src_dir.exists() and str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import uvicorn
from rag.config import settings

if __name__ == "__main__":
    print(f"Starting backend server on {settings.api_host}:{settings.api_port}...")
    uvicorn.run("rag.api.app:app", host=settings.api_host, port=settings.api_port, reload=True)

