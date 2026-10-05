import importlib.util
import tempfile
import unittest
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("packaging_check", ROOT / "tools/verify_spectral_apk.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class SpectralPackagingTests(unittest.TestCase):
    def make_apk(self, path, extra=()):
        with zipfile.ZipFile(path, "w") as apk:
            for name in ("AndroidManifest.xml", "classes.dex", *extra):
                apk.writestr(name, b"synthetic")

    def test_lightweight_apk_passes(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "test.apk"
            self.make_apk(path)
            self.assertEqual(2, module.verify(path))

    def test_model_runtime_and_recordings_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "test.apk"
            for entry in ("assets/models/yamnet.tflite", "lib/arm64-v8a/libtensorflowlite_jni.so",
                          "assets/audio_clips/night.m4a", "assets/private-analysis/labels.json"):
                self.make_apk(path, [entry])
                with self.assertRaises(ValueError):
                    module.verify(path)

    def test_regular_zip_is_not_an_apk(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "test.apk"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("notes.txt", "not an APK")
            with self.assertRaises(ValueError):
                module.verify(path)


if __name__ == "__main__":
    unittest.main()
