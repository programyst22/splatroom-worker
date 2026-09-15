import argparse
import importlib
import shutil
import subprocess
import sys

REQUIRED_COMMANDS = ["ffmpeg", "colmap", "ns-process-data", "ns-train", "ns-export"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-check", action="store_true")
    args = parser.parse_args()

    missing = [cmd for cmd in REQUIRED_COMMANDS if shutil.which(cmd) is None]
    if missing:
        raise SystemExit(f"Missing required commands: {', '.join(missing)}")

    for module in ("torch", "gsplat"):
        importlib.import_module(module)

    # During Docker build there is no GPU. Runtime verifies CUDA separately.
    if args.build_check:
        print("SPLATROOM build preflight OK")
        return

    import torch
    print("Torch:", torch.__version__)
    print("CUDA build:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU is not available")
    print("GPU:", torch.cuda.get_device_name(0))
    print(subprocess.check_output(["colmap", "-h"], text=True, stderr=subprocess.STDOUT).splitlines()[0])
    print("SPLATROOM runtime preflight OK")


if __name__ == "__main__":
    main()
