#!/usr/bin/env python3
"""
ctx_bootstrap.py - installs or upgrades ctx_memory into the active venv
and writes a `ctx` wrapper into /usr/local/bin/.
Run as:   python3 /home/ayan/context_memory/src/ctx_memory/ctx_bootstrap.py
"""
import subprocess
import sys
import os
from pathlib import Path

CTX_DIR = Path(__file__).resolve().parent
CTX_WRAPPER = """#!/usr/bin/env bash
# ctx - Hermes local context memory
CTX_DIR="__CTX_DIR__"
exec python3 "$CTX_DIR/cli.py" "$@"
"""


def run(cmd, check=True):
    print(f">>> {cmd}")
    return subprocess.run(cmd, shell=True, check=check)


def main():
    if sys.prefix == sys.base_prefix:
        print("INFO: not inside a venv. Using system python3.")
        python = "python3"
    else:
        print(f"INFO: venv detected at {sys.prefix}")
        python = sys.executable

    print("Installing numpy...")
    run(f"{python} -m pip install numpy --quiet", check=False)

    print("Installing scikit-learn (optional but speeds embeddings)...")
    run(f"{python} -m pip install scikit-learn --quiet", check=False)

    wrapper_src = CTX_WRAPPER.replace("__CTX_DIR__", str(CTX_DIR))
    wrapper_path = "/usr/local/bin/ctx"
    tmp = wrapper_path + ".tmp"
    with open(tmp, 'w') as f:
        f.write(wrapper_src)
    os.chmod(tmp, 0o755)
    os.replace(tmp, wrapper_path)
    print(f"Installed `ctx` wrapper to {wrapper_path}")


if __name__ == "__main__":
    main()
