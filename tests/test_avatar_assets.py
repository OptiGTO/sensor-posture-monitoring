from __future__ import annotations

import json
import unittest
from pathlib import Path

from src.avatar_assets import (
    POSTURE_AVATAR_FILENAMES,
    WAITING_AVATAR_FILENAME,
    avatar_filename,
    missing_avatar_assets,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSET_DIRECTORY = PROJECT_ROOT / "software" / "assets" / "postures"


class AvatarAssetTests(unittest.TestCase):
    def test_deployed_postures_have_distinct_png_assets(self) -> None:
        manifest = json.loads(
            (PROJECT_ROOT / "artifacts" / "manifests" / "posture_classifier_3s.json").read_text(encoding="utf-8")
        )
        labels = tuple(map(str, manifest["labels"]))

        self.assertEqual(set(POSTURE_AVATAR_FILENAMES), set(labels))
        self.assertEqual(len(set(POSTURE_AVATAR_FILENAMES.values())), len(labels))
        self.assertEqual(missing_avatar_assets(ASSET_DIRECTORY), ())
        for filename in (*POSTURE_AVATAR_FILENAMES.values(), WAITING_AVATAR_FILENAME):
            self.assertTrue((ASSET_DIRECTORY / filename).read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))

    def test_unknown_or_empty_posture_uses_waiting_avatar(self) -> None:
        self.assertEqual(avatar_filename(None), WAITING_AVATAR_FILENAME)
        self.assertEqual(avatar_filename("판단 유보"), WAITING_AVATAR_FILENAME)


if __name__ == "__main__":
    unittest.main()
