from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent.parent


DATA_DIR = PROJECT_ROOT / "data"
BITGET_DATA_DIR = DATA_DIR / "bitget"


OPTIMIZER_DIR = PROJECT_ROOT / "optimizer_studies"


QUANTSSTATS_DIR = PROJECT_ROOT / "quantstats"

ANALYSIS_DIR = PROJECT_ROOT / "analyze_study"


DATA_DIR.mkdir(exist_ok=True)
OPTIMIZER_DIR.mkdir(exist_ok=True)
QUANTSSTATS_DIR.mkdir(exist_ok=True)
ANALYSIS_DIR.mkdir(exist_ok=True)
BITGET_DATA_DIR.mkdir(exist_ok=True) 