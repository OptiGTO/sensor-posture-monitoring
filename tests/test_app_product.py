from __future__ import annotations

import json
import tempfile
import unittest
import os
from unittest.mock import patch
from pathlib import Path

import build_macos_app
from src.app_products import GTO_PRODUCT, PRODUCTS, select_product, PRODUCT_ENVIRONMENT_VARIABLE


class AppProductTests(unittest.TestCase):
    def test_gto_identity_and_existing_user_state_are_preserved(self) -> None:
        self.assertEqual(set(PRODUCTS), {"gto"})
        self.assertEqual(GTO_PRODUCT.owner_name, "안재현")
        self.assertEqual(GTO_PRODUCT.output_directory_prefix, "안재현-gto")
        self.assertEqual(GTO_PRODUCT.display_name, "자세 측정 및 관리")
        self.assertEqual(GTO_PRODUCT.executable_name, "PostureMonitor")
        self.assertEqual(GTO_PRODUCT.bundle_identifier, "kr.ac.hufs.bioacoustics.posture-monitor")
        self.assertEqual(GTO_PRODUCT.entrypoint, "software/posture_app.py")
        self.assertEqual(GTO_PRODUCT.state_root(Path("/tmp/user")),
                         Path("/tmp/user/Library/Application Support/자세 측정 및 관리"))

    def test_selection_defaults_to_gto_and_rejects_unknown_variants(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(select_product(), GTO_PRODUCT)
        self.assertEqual(select_product("gto"), GTO_PRODUCT)
        with patch.dict(os.environ, {PRODUCT_ENVIRONMENT_VARIABLE: "removed-product"}):
            with self.assertRaises(ValueError):
                select_product()

    def test_build_target_keeps_fixed_entrypoint_and_bundle_id(self) -> None:
        command = build_macos_app.build_command()
        self.assertEqual(Path(command[-1]), build_macos_app.ROOT / "software/posture_app.py")
        self.assertEqual(command[command.index("--name") + 1], "PostureMonitor")
        self.assertEqual(command[command.index("--osx-bundle-identifier") + 1],
                         "kr.ac.hufs.bioacoustics.posture-monitor")

    def test_build_identity_is_packaged_at_the_resource_root(self) -> None:
        identity = Path("/tmp/build_identity.json")
        command = build_macos_app.build_command(identity_path=identity)
        add_data = [
            command[index + 1]
            for index, value in enumerate(command[:-1])
            if value == "--add-data"
        ]
        self.assertIn(f"{identity}:.", add_data)
        self.assertNotIn(f"{identity}:build_identity.json", add_data)

    def test_build_identity_records_exact_product_and_entrypoint_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "build_identity.json"
            identity = build_macos_app.write_build_identity(path, GTO_PRODUCT)
            persisted = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(identity, persisted)
        self.assertEqual(identity["product_variant"], "gto")
        self.assertEqual(identity["owner_name"], "안재현")
        self.assertEqual(identity["output_directory_prefix"], "안재현-gto")
        self.assertEqual(identity["entrypoint"], GTO_PRODUCT.entrypoint)
        self.assertEqual(len(identity["entrypoint_sha256"]), 64)

    def test_build_directories_must_include_the_owner_prefix(self) -> None:
        build_macos_app.require_product_labeled_directory(
            Path("/tmp/안재현-gto-release"), GTO_PRODUCT, "--distpath",
        )
        for name in ("release", "wrong-owner-release", "안재현-gto-"):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "안재현"):
                build_macos_app.require_product_labeled_directory(Path("/tmp") / name, GTO_PRODUCT, "--distpath")


if __name__ == "__main__":
    unittest.main()
