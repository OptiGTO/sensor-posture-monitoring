#!/usr/bin/env python3
"""Validate a model manifest without loading its model weights."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.model_runtime.validation import ArtifactValidator


def main() -> int:
    parser = argparse.ArgumentParser(description="자세 모델 아티팩트 매니페스트를 검증합니다.")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--allow-experimental", action="store_true")
    args = parser.parse_args()
    result = ArtifactValidator().validate_manifest(args.manifest, production_only=not args.allow_experimental)
    if not result.available:
        print(f"검증 실패: {result.reason}")
        return 1
    descriptor = result.descriptor
    assert descriptor is not None
    print(f"검증 통과: {descriptor.display_name} ({descriptor.artifact_id}, {descriptor.window_label})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
