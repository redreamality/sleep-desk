"""Create reproducible synthetic-only Kotlin/Python parity fixtures, never private recordings."""
import argparse
import json
from pathlib import Path

import numpy as np

from audio_feature_study import extract_features

CASES = [
    ("silence", 512, 0), ("constant", 800, 1), ("triangle", 15600, 2),
    ("noise", 16000, 3), ("burst", 48000, 4), ("contact", 27637, 5),
    ("short", 127, 4), ("odd_burst", 32767, 4), ("full_context", 192000, 4),
]


def synthetic_pcm(size, kind):
    state = 17
    values = []
    for i in range(size):
        state = (state * 1664525 + 1013904223) & 0xffffffff
        noise = ((state >> 16) & 65535) - 32768
        triangle = (abs(i % 160 - 80) - 40) * 200
        if kind == 0:
            value = 0
        elif kind == 1:
            value = 111
        elif kind == 2:
            value = triangle
        elif kind == 3:
            value = int(noise / 8)
        elif kind == 4:
            value = int(triangle / 2) + int(noise / 32) if i % 48000 in range(16000, 32000) else int(noise / 512)
        else:
            value = int(noise / 128) + (int(15000 / (i % 8000 + 1)) if i % 8000 < 64 else 0)
        values.append(value)
    return np.array(values, dtype=np.int16)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parameters", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model = json.loads(args.parameters.read_text(encoding="utf-8-sig"))
    names = model["feature_names"]
    lines = ["name\tsize\tkind\tmargin\t" + "\t".join(names)]
    for name, size, kind in CASES:
        y = synthetic_pcm(size, kind).astype(float) / 32768
        f = extract_features(y)
        values = [f[n.removeprefix("hand.")] for n in names]
        margin = float(np.array(values) @ np.array(model["weights"]) + model["bias"])
        lines.append("\t".join([name, str(size), str(kind), repr(margin)] + [repr(v) for v in values]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Wrote synthetic-only feature fixtures:", len(CASES))


if __name__ == "__main__":
    main()
