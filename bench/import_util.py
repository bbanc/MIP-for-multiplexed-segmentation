import os
from pathlib import Path


DATA_ROOT = Path("/PATH/TO/DATA/DIR") 
CELLVIT_ROOT = Path(os.environ.get("CELLVIT_ROOT", str("PATH/TO/CELLVIT/CLONE"))) # Manual import Path to Cellvit git dir (no pip install)

def data_path(*parts: str) -> Path:
    return DATA_ROOT / Path(*parts)

def cellvit_path(*parts: str) -> Path:
    return CELLVIT_ROOT / Path(*parts)
