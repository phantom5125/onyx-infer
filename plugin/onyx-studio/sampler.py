"""Live process sampler for Onyx Studio generation jobs.

Watches one ``onyx.replay`` process tree, printing one NDJSON object per tick:

    {"t": 1690000000.0, "rss": 1234567890, "stage": "dit"}

``stage`` is derived from the ``onyx-native <stage>`` child command line, so
the panel can show text/DiT/VAE progress live. Uses the engine venv's psutil.
Exits when the watched process disappears or turns into a zombie.
"""
import json
import sys
import time

try:
    import psutil
except ImportError:  # pragma: no cover - the engine venv always has psutil
    print(json.dumps({"fatal": "psutil unavailable"}), flush=True)
    sys.exit(1)


def main() -> int:
    if len(sys.argv) < 2:
        print(json.dumps({"fatal": "usage: sampler.py <pid> [interval-seconds]"}), flush=True)
        return 2
    pid = int(sys.argv[1])
    interval = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        print(json.dumps({"t": time.time(), "rss": 0, "stage": "gone"}), flush=True)
        return 0
    while True:
        procs = [parent]
        rss = 0
        stage = None
        try:
            procs += parent.children(recursive=True)
        except psutil.Error:
            pass
        for proc in procs:
            try:
                rss += proc.memory_info().rss
                if stage is None:
                    cmdline = proc.cmdline()
                    if len(cmdline) >= 2 and cmdline[0].endswith("onyx-native"):
                        stage = cmdline[1]
            except psutil.Error:
                continue
        if rss > 0 or stage is not None:
            print(json.dumps({"t": round(time.time(), 3), "rss": rss, "stage": stage}), flush=True)
        try:
            if not parent.is_running() or parent.status() == psutil.STATUS_ZOMBIE:
                break
        except psutil.Error:
            break
        time.sleep(interval)
    return 0


if __name__ == "__main__":
    sys.exit(main())
