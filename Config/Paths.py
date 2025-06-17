from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent.parent


DATA_DIR = PROJECT_ROOT / "data"


OPTIMIZER_DIR = PROJECT_ROOT / "optimizer_studies"


QUANTSSTATS_DIR = PROJECT_ROOT / "quantstats"


DATA_DIR.mkdir(exist_ok=True)
OPTIMIZER_DIR.mkdir(exist_ok=True)
QUANTSSTATS_DIR.mkdir(exist_ok=True)