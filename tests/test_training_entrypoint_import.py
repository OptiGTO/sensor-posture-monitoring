"""Non-plotting workers must not scan system fonts before training starts."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class TrainingEntrypointImportTests(unittest.TestCase):
    def test_cold_entrypoint_import_does_not_load_matplotlib(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix="posture-cold-font-cache-") as temporary:
            environment = os.environ.copy()
            environment["MPLCONFIGDIR"] = temporary
            probe = (
                "import sys; import train_window_models; "
                "assert not any(name.startswith('matplotlib') for name in sys.modules)"
            )
            result = subprocess.run([sys.executable, "-c", probe], cwd=root, env=environment,
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(list(Path(temporary).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
