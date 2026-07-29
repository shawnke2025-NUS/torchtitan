#!/usr/bin/env python3
#version2:12:43
from __future__ import annotations

import argparse
import py_compile
import shutil
from datetime import datetime
from pathlib import Path

OLD = """    if parallel_dims.tp_enabled or parallel_dims.ep_enabled:\n"""
NEW = """    # Qwen3.5 sharding configs are expressed with spmd_types.  They must be\n    # applied even when tensor_parallel_degree == 1; otherwise PP/FSDP can\n    # produce DTensor activations while norm/projection parameters remain plain\n    # torch.Tensor objects, causing mixed Tensor/DTensor operator failures.\n    if (\n        parallelism.spmd_backend == \"spmd_types\"\n        or parallel_dims.tp_enabled\n        or parallel_dims.ep_enabled\n    ):\n"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("repo", nargs="?", default=".", help="TorchTitan repository root")
    args = parser.parse_args()

    repo = Path(args.repo).expanduser().resolve()
    path = repo / "torchtitan/models/qwen3_5/parallelize.py"
    if not path.is_file():
        raise SystemExit(f"File not found: {path}")

    text = path.read_text(encoding="utf-8")
    if NEW in text:
        print(f"Already patched: {path}")
    else:
        count = text.count(OLD)
        if count != 1:
            raise SystemExit(
                f"Expected exactly one target condition, found {count}. "
                "Inspect parallelize.py manually before modifying it."
            )
        backup = path.with_suffix(
            path.suffix + ".before_tp1_pp_fix_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        shutil.copy2(path, backup)
        path.write_text(text.replace(OLD, NEW, 1), encoding="utf-8")
        print(f"Backup: {backup}")
        print(f"Patched: {path}")

    py_compile.compile(str(path), doraise=True)
    print("py_compile: OK")

    updated = path.read_text(encoding="utf-8").splitlines()
    for idx, line in enumerate(updated, start=1):
        if 'parallelism.spmd_backend == "spmd_types"' in line:
            start = max(1, idx - 5)
            end = min(len(updated), idx + 8)
            print("\nPatched section:")
            for lineno in range(start, end + 1):
                print(f"{lineno:4d}: {updated[lineno - 1]}")
            break


if __name__ == "__main__":
    main()
