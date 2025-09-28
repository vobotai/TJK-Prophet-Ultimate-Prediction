from __future__ import annotations

import sys
from pathlib import Path

_src_path = Path(__file__).resolve().parents[1] / "src"
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))
