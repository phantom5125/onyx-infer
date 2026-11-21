import sys


def main() -> None:
    commands = {
        "import": "onyx.importer",
        "quantize": "onyx.quantize",
        "replay": "onyx.replay",
        "metal-record": "onyx.metal",
        "export-metal-fixture": "onyx.fixture",
        "package-w4": "onyx.w4_package",
        "native-generate": "onyx.native",
        "compose-labels": "onyx.labels",
        "label-prompt": "onyx.label_prompt",
    }
    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        raise SystemExit(f"usage: onyx <{'|'.join(commands)}> [options]")
    module = __import__(commands[sys.argv[1]], fromlist=["main"])
    sys.argv = [f"onyx {sys.argv[1]}", *sys.argv[2:]]
    module.main()


if __name__ == "__main__":
    main()
