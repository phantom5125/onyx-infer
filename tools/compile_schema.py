from pathlib import Path

from grpc_tools import protoc


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "schema"
OUT = ROOT / "src"
BUILD = ROOT / ".build" / "schema"


def main() -> None:
    files = sorted(str(path.relative_to(SCHEMA)) for path in SCHEMA.rglob("*.proto"))
    BUILD.mkdir(parents=True, exist_ok=True)
    args = [
        "protoc",
        f"-I{SCHEMA}",
        f"--python_out={OUT}",
        f"--descriptor_set_out={BUILD / 'compiled.pb'}",
        "--include_imports",
        *files,
    ]
    if protoc.main(args) != 0:
        raise SystemExit("Protocol compilation failed")
    for path in (OUT / "onyx").rglob("*"):
        if path.is_dir():
            (path / "__init__.py").touch()
    print(f"Compiled {len(files)} protocol files")


if __name__ == "__main__":
    main()
