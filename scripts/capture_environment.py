#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path


def command_output(command: list[str]) -> str:
    try:
        result = subprocess.run(command, text=True, capture_output=True, check=False, timeout=30)
        return (result.stdout + result.stderr).strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"unavailable: {exc}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/environment.json")
    args = parser.parse_args()
    payload = {
        "python": sys.version,
        "platform": platform.platform(),
        "git_commit": command_output(["git", "rev-parse", "HEAD"]),
        "git_status": command_output(["git", "status", "--short"]),
        "nvidia_smi": command_output(["nvidia-smi"]),
        "gpu_topology": command_output(["nvidia-smi", "topo", "-m"]),
        "pip_freeze": command_output([sys.executable, "-m", "pip", "freeze"]).splitlines(),
    }
    try:
        import torch

        payload["torch"] = {
            "version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "devices": [
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "total_memory": torch.cuda.get_device_properties(index).total_memory,
                }
                for index in range(torch.cuda.device_count())
            ],
        }
    except ImportError:
        payload["torch"] = "not installed"
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
