"""Export only numeric linear-model parameters; never include private training provenance."""
import argparse
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parameters", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model = json.loads(args.parameters.read_text(encoding="utf-8-sig"))
    names, weights = model["feature_names"], model["weights"]
    if model["format"] != "spectral-linear-v1" or len(names) != 48 or len(set(names)) != 48:
        raise ValueError("Expected frozen 48-feature linear classifier")
    if len(weights) != len(names) or not all(math.isfinite(x) for x in weights + [model["bias"]]):
        raise ValueError("Invalid weights")
    if any(not n.startswith(("hand.env.", "hand.spec.", "hand.dyn.")) for n in names):
        raise ValueError("Unsupported feature namespace")
    lines = [
        "package com.i3u8.sleepdesk.audio", "",
        "/** Generated numerical parameters only; no training audio, labels or source identities. */",
        "internal object SpectralParameters {",
        f'    const val BIAS = {model["bias"]!r}',
        f'    const val THRESHOLD = {model["threshold"]!r}',
        "    val names = arrayOf(",
        ",\n".join("        " + json.dumps(n) for n in names),
        "    )",
        "    val weights = doubleArrayOf(",
        ",\n".join("        " + repr(w) for w in weights),
        "    )", "}", "",
    ]
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Exported {len(weights)} weights without private metadata")


if __name__ == "__main__":
    main()
