#!/usr/bin/env python3
"""TEST ONLY: isolated executable fixture; never a production engine or dataset."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

HELP = """TEST ONLY llama-bench fixture
 -m, --model path
 -p, --n-prompt n
 -n, --n-gen n
 -t, --threads n
 -ngl, --n-gpu-layers n
 -b, --batch-size n
 -ub, --ubatch-size n
 -fa, --flash-attn <on|off|auto>
 -r, --repetitions n
 -o, --output <json>
 -dev, --device name
 -nkvo, --no-kv-offload n
 -nopo, --no-op-offload n
 --offline
 --list-devices
"""


def main() -> int:
    args = sys.argv[1:]
    if "--help" in args:
        print(HELP)
        return 0
    if "--version" in args:
        print("TEST ONLY llama.cpp fixture commit 000000000")
        return 0
    if "--list-devices" in args:
        print("Available devices:\n  MTL0: TEST ONLY synthetic device")
        return 0
    marker = os.environ.get("LIOL_TEST_READY_FILE")
    if marker:
        Path(marker).write_text(str(os.getpid()), encoding="utf-8")
    behavior = os.environ.get("LIOL_TEST_BEHAVIOR", "success")
    if behavior == "failure":
        print("TEST ONLY deliberate engine failure", file=sys.stderr, flush=True)
        return 7
    if behavior == "wait":
        print("TEST ONLY process started", flush=True)
        time.sleep(30)
    if behavior == "malformed":
        print("TEST ONLY not valid JSON")
        return 0

    def number(flag: str) -> int:
        return int(args[args.index(flag) + 1])

    count = number("-r")
    prompt, generated = number("-p"), number("-n")
    durations = [1_000_000 + i * 100_000 for i in range(count)]
    row = {
        "TEST_ONLY": True,
        "build_commit": "000000000",
        "build_number": 0,
        "model_filename": args[args.index("-m") + 1],
        "n_prompt": prompt,
        "n_gen": generated,
        "n_threads": number("-t"),
        "n_gpu_layers": number("-ngl"),
        "n_batch": number("-b"),
        "n_ubatch": number("-ub"),
        "n_depth": 0,
        "flash_attn": {"auto": -1, "off": 0, "on": 1}[args[args.index("-fa") + 1]],
        "no_kv_offload": number("-nkvo") == 1,
        "no_op_offload": number("-nopo"),
        "devices": args[args.index("-dev") + 1],
        "samples_ns": durations,
        "samples_ts": [(prompt + generated) * 1e9 / duration for duration in durations],
    }
    print(json.dumps([row]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
