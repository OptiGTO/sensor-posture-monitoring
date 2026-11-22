"""Stable identity for the GTO desktop application."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PRODUCT_ENVIRONMENT_VARIABLE = "POSTURE_APP_VARIANT"


@dataclass(frozen=True)
class AppProduct:
    variant: str
    owner_name: str
    output_directory_prefix: str
    display_name: str
    executable_name: str
    bundle_identifier: str
    state_directory_name: str
    entrypoint: str

    def state_root(self, home: Path | None = None) -> Path:
        home = Path.home() if home is None else Path(home)
        return home / "Library" / "Application Support" / self.state_directory_name


GTO_PRODUCT = AppProduct(
    variant="gto",
    owner_name="안재현",
    output_directory_prefix="안재현-gto",
    display_name="자세 측정 및 관리",
    executable_name="PostureMonitor",
    bundle_identifier="kr.ac.hufs.bioacoustics.posture-monitor",
    state_directory_name="자세 측정 및 관리",
    entrypoint="software/posture_app.py",
)

PRODUCTS = {GTO_PRODUCT.variant: GTO_PRODUCT}


def select_product(variant: str | None = None) -> AppProduct:
    selected = variant or os.environ.get(PRODUCT_ENVIRONMENT_VARIABLE, GTO_PRODUCT.variant)
    try:
        return PRODUCTS[selected]
    except KeyError as error:
        choices = ", ".join(sorted(PRODUCTS))
        raise ValueError(f"Unknown posture app variant {selected!r}; expected one of: {choices}") from error


def force_product(product: AppProduct) -> None:
    """Pin the process before importing the shared Tk application module."""

    os.environ[PRODUCT_ENVIRONMENT_VARIABLE] = product.variant
