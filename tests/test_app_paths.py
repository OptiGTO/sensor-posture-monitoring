from pathlib import Path
import tempfile
from unittest import TestCase, mock

from src.app_paths import data_directory, dataset_directories, writable_data_directory
from src.app_services import DatasetService, MeasurementRecorder


class AppPathTests(TestCase):
    def test_listing_includes_legacy_data_without_changing_recording_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            legacy = root / 'data'
            legacy.mkdir()
            recordings = root / 'gto-state/data'
            for executable in (None, root / 'dist/안재현-gto-test/자세 측정 및 관리.app/Contents/MacOS/PostureMonitor'):
                with self.subTest(executable=executable):
                    self.assertEqual(dataset_directories(recordings, root, executable=executable),
                                     (recordings, legacy))
                    self.assertEqual(dataset_directories(recordings, root, executable=executable, isolated=True),
                                     (recordings,))
            self.assertFalse(recordings.exists())
            self.assertEqual(dataset_directories(legacy, root), (legacy,))

    def test_listing_does_not_scan_unrelated_state_or_missing_portable_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            other = root / 'unrelated-state/data'
            other.mkdir(parents=True)
            recordings = root / 'gto-state/data'
            self.assertEqual(dataset_directories(recordings, root), (recordings,))

    def test_writable_directory_needs_no_deletable_probe_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / 'data'
            with mock.patch('tempfile.TemporaryFile', side_effect=PermissionError('delete denied')):
                self.assertEqual(writable_data_directory(data, root / 'state'), data)
            self.assertEqual(list(data.iterdir()), [])

    def test_read_only_directory_uses_writable_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with mock.patch('src.app_paths.os.access', side_effect=[False, True]):
                self.assertEqual(writable_data_directory(root / 'data', root / 'state'), root / 'state/data')

    def test_source_and_packaged_layouts_share_project_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            expected = root / 'data'
            self.assertEqual(data_directory(root), expected)
            for relative in ('dist/자세 측정 및 관리.app/Contents/MacOS/app',
                             'dist/자세 측정 및 관리/app'):
                with self.subTest(layout=relative):
                    actual = data_directory(root / 'bundle-resources', executable=root / relative)
                    self.assertEqual(actual, expected)
                    recorder = MeasurementRecorder(root / 'state', mode='measurement', data_directory=actual)
                    recorder.write_frame(0, range(16))
                    recorder.close()
                    records = DatasetService(root / 'state', data_directory=data_directory(root)).records()
                    self.assertIn(recorder.csv_path, [record.csv_path for record in records])

    def test_moved_app_uses_data_beside_app(self):
        root = Path('/tmp/portable').resolve()
        self.assertEqual(data_directory(root / 'resources', executable=root / 'Posture.app/Contents/MacOS/app'), root / 'data')
