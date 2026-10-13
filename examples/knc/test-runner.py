#!/usr/bin/env python3
"""Run bounded local fixture checks. Does not connect to or configure a card."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

import numpy as np


def main():
    if not __debug__:
        raise RuntimeError("Run these checks without Python -O/PYTHONOPTIMIZE")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runner", required=True, type=Path)
    p.add_argument("--fixtures", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--timeout", type=int, default=120)
    p.add_argument("--native-guarded", action="store_true", help="Native invocation is supervised by an external affinity/thermal guard")
    p.add_argument("--reference", type=Path, help="Host-reference results for the identical fixture manifest")
    a = p.parse_args()
    manifest = json.loads((a.fixtures / "manifest.json").read_text())
    for name, entry in manifest["files"].items():
        assert hashlib.sha256((a.fixtures / name).read_bytes()).hexdigest() == entry["sha256"], name
    identity = subprocess.check_output([str(a.runner), "--describe"], text=True, timeout=10)
    if json.loads(identity)["profile"] != "HOST_REFERENCE" and (not a.native_guarded or not a.reference):
        p.error("Native tests require the external hardware guard, --native-guarded and --reference")
    fixture_hash = hashlib.sha256((a.fixtures / "manifest.json").read_bytes()).hexdigest()
    if a.reference:
        ref = json.loads((a.reference / "results.json").read_text())
        assert ref["fixture_manifest_sha256"] == fixture_hash, "Reference fixtures differ"
        assert ref["identity"]["profile"] == "HOST_REFERENCE", "Expected a host reference"
    a.output.mkdir(parents=True, exist_ok=False)
    results = []

    def run(name, model="qwen3-Q4_K_M.gguf", prompt="text.txt", extra=(), error=None):
        cmd = [str(a.runner), "-m", str(a.fixtures / model), "-f", str(a.fixtures / prompt),
               "-t", str(a.threads), "-tb", str(a.threads), "-c", "256", "-b", "16", "-n", "4", *extra]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=a.timeout)
        (a.output / (name + ".stdout")).write_bytes(proc.stdout)
        (a.output / (name + ".stderr")).write_bytes(proc.stderr)
        log = proc.stderr.decode("utf-8", errors="replace")
        if error:
            assert proc.returncode != 0 and error in log, (name, proc.returncode, log[-2000:])
        else:
            assert proc.returncode == 0, (name, proc.returncode, log[-3000:])
        results.append({"case": name, "returncode": proc.returncode, "expected_rejection": bool(error)})
        return log

    for arch in ("llama", "lfm2", "lfm2moe", "qwen3"):
        for quant in ("Q3_K_M", "Q4_K_M", "Q5_K_M"):
            name = f"{arch}-{quant}"
            dump = a.output / name
            dump.mkdir()
            log = run(name + "-state", name + ".gguf", extra=["--validate-state", str(dump), "--cycles", "2"])
            assert log.count("PHI_NUMERICS") == 6
            log = run(name + "-generate", name + ".gguf")
            metrics = json.loads(re.search(r"PHI_METRICS (.*)", log)[1])
            assert metrics["generated_tokens"] > 0

    mm = ["--mmproj", str(a.fixtures / "step3vl-mmproj.gguf")]
    for quant in ("Q3_K_M", "Q4_K_M", "Q5_K_M"):
        logits = []
        for color in ("red", "blue"):
            name = "image-" + quant + "-" + color
            dump = a.output / name
            dump.mkdir()
            image = [*mm, "--image", str(a.fixtures / (color + ".png"))]
            log = run(name, f"qwen3-{quant}.gguf", "image.txt", [*image, "--validate-state", str(dump)])
            assert "PHI_NUMERICS image_reset_vs_fresh" in log
            logits.append(np.fromfile(dump / "cycle-0-image.f32", dtype=np.float32))
            run(name + "-generate", f"qwen3-{quant}.gguf", "image.txt", image)
        assert np.isfinite(logits).all()
        assert np.max(np.abs(logits[0] - logits[1])) > 1e-6, "Image pixels did not affect logits"

    run("missing-projector", extra=["--image", str(a.fixtures / "red.png")], error="both --mmproj and --image")
    run("marker-without-image", prompt="image.txt", error="Media marker requires image input")
    image = [*mm, "--image", str(a.fixtures / "red.png")]
    run("missing-marker", extra=image, error="one media marker per image")
    run("too-many-images", extra=[*mm, *sum((["--image", str(a.fixtures / "red.png")] for _ in range(5)), [])], error="At most four")
    run("context-overflow", prompt="image.txt", extra=[*image, "-c", "128", "-n", "127"], error="Expanded prompt plus reply exceeds")
    run("not-an-image", prompt="image.txt", extra=[*mm, "--image", str(a.fixtures / "text.txt")], error="Expected a supported still image")
    run("mmap-required", extra=["--no-mmap"], error="Shared mmap required")
    run("conflicting-kernels", extra=["--baseline-kernels", "--scalar-kernels"], error="Choose only one")
    comparisons = []
    if a.reference:
        for path in sorted(a.output.rglob("*.f32")):
            local = np.fromfile(path, dtype=np.float32).astype(np.float64)
            reference = np.fromfile(a.reference / path.relative_to(a.output), dtype=np.float32).astype(np.float64)
            assert local.shape == reference.shape and np.isfinite(local).all() and np.isfinite(reference).all()
            diff = np.abs(local - reference)
            nr = float(np.sqrt(np.sum(diff * diff) / max(np.sum(reference * reference), 1e-30)))
            assert nr <= 0.002 and np.all(diff <= 0.03 + 0.005 * np.abs(reference)), str(path)
            comparisons.append({"file": str(path.relative_to(a.output)), "nrmse": nr, "max_abs": float(diff.max())})
    receipt = {"identity": json.loads(identity), "runner_sha256": hashlib.sha256(a.runner.read_bytes()).hexdigest(),
               "fixture_manifest_sha256": fixture_hash, "reference_comparisons": comparisons,
               "scope": "synthetic correctness only; not trained-model quality or native throughput", "cases": results}
    (a.output / "results.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"PASS: {len(results)} cases; see {a.output / 'results.json'}")


if __name__ == "__main__":
    main()
