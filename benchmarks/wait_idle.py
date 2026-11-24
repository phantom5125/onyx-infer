"""Run a command once no native stage process from another run is active.

    uv run python benchmarks/wait_idle.py -- <command...>

The GPU is shared with other sessions on the same machine, and a sample or a probe
that overlaps another native stage is not comparable. The command runs when no
process named onyx-native is executing a text, dit, or vae stage, after an
optional extra settle delay.
"""

import argparse
import subprocess
import sys
import time

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from ab_compare import native_stage_pids  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--settle", type=float, default=2.0, help="Seconds of quiet time before the command starts")
    parser.add_argument("--timeout", type=float, default=3600, help="Give up waiting after this many seconds")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command and args.command[0] == "--" else args.command
    if not command:
        parser.error("pass a command after --")
    began = time.monotonic()
    quiet_since = None
    while True:
        busy = native_stage_pids()
        if busy:
            quiet_since = None
            if time.monotonic() - began > args.timeout:
                sys.exit(f"gave up waiting: native stage processes {busy} still run")
            print(f"waiting for native stage processes {busy}", flush=True)
            time.sleep(5)
            continue
        quiet_since = quiet_since or time.monotonic()
        if time.monotonic() - quiet_since >= args.settle:
            break
        time.sleep(0.5)
    if time.monotonic() - began > args.settle:
        print(f"idle after {time.monotonic() - began:.0f} s", flush=True)
    result = subprocess.run(command)
    still = native_stage_pids()
    if still:
        print(f"note: native stage processes {still} were active when the command ended", flush=True)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
