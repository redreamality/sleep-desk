"""Reject production APKs that bundle the historical large model or private recordings."""
import argparse
import zipfile
from pathlib import Path


def verify(path):
    with zipfile.ZipFile(path) as apk:
        names = apk.namelist()
        if "AndroidManifest.xml" not in names or "classes.dex" not in names:
            raise ValueError("Not an Android application APK")
        prohibited = [name for name in names if (
            name.endswith((".tflite", ".pcm", ".m4a", ".wav"))
            or "private-analysis" in name or "audio_clips/" in name
            or "tensorflow" in name.lower()
        )]
        if prohibited:
            raise ValueError("Unexpected model, runtime or recording assets: " + ", ".join(prohibited))
    return len(names)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("apk", type=Path)
    args = parser.parse_args()
    print(f"Spectral APK packaging verified: {verify(args.apk)} entries, no TFLite or recordings.")
