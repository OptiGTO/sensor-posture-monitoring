#!/usr/bin/env python3
"""GTO desktop product entrypoint.

This wrapper pins the GTO identity before the application module is imported,
preserving the established bundle identity and runtime state location.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.app_products import GTO_PRODUCT, force_product

force_product(GTO_PRODUCT)

from software import posture_app_base as _base  # noqa: E402


if __name__ == "__main__":
    _base.run_application()
else:
    # Preserve the historical import and monkeypatch surface. Callers importing
    # software.posture_app receive the real implementation module, not a proxy.
    sys.modules[__name__] = _base
