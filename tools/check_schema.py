import argparse
import json
from pathlib import Path

from google.protobuf import descriptor_pb2


ROOT = Path(__file__).resolve().parents[1]
DESCRIPTOR = ROOT / ".build" / "schema" / "compiled.pb"
BASELINE = ROOT / "schema" / "v1.compatibility.json"


def snapshot():
    descriptor = descriptor_pb2.FileDescriptorSet.FromString(DESCRIPTOR.read_bytes())
    messages = {}
    enums = {}

    def visit_message(message, prefix):
        name = f"{prefix}.{message.name}"
        messages[name] = {
            field.name: {
                "number": field.number,
                "label": field.label,
                "type": field.type,
                "type_name": field.type_name,
            }
            for field in message.field
        }
        visit_enum(message.enum_type, name)
        for nested in message.nested_type:
            visit_message(nested, name)

    def visit_enum(values, prefix):
        for value in values:
            name = f"{prefix}.{value.name}"
            enums[name] = {item.name: item.number for item in value.value}

    for file in descriptor.file:
        visit_enum(file.enum_type, file.package)
        for message in file.message_type:
            visit_message(message, file.package)
    return {"messages": messages, "enums": enums}


def changes(baseline, current, kind):
    problems = []
    additions = []
    for name, fields in baseline[kind].items():
        if name not in current[kind]:
            problems.append(f"removed {kind[:-1]}: {name}")
            continue
        for field, expected in fields.items():
            if field not in current[kind][name]:
                problems.append(f"removed {name}.{field}")
            elif current[kind][name][field] != expected:
                problems.append(f"changed {name}.{field}: {expected} != {current[kind][name][field]}")
    for name, fields in current[kind].items():
        old = baseline[kind].get(name, {})
        if name not in baseline[kind]:
            additions.append(f"added {kind[:-1]}: {name}")
        for field in fields:
            if field not in old:
                additions.append(f"added {name}.{field}")
    return problems, additions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-baseline", action="store_true")
    args = parser.parse_args()
    current = snapshot()
    if args.write_baseline:
        BASELINE.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")
        print(f"Wrote {BASELINE.relative_to(ROOT)}")
        return
    baseline = json.loads(BASELINE.read_text())
    problems = []
    additions = []
    for kind in ("messages", "enums"):
        new_problems, new_additions = changes(baseline, current, kind)
        problems.extend(new_problems)
        additions.extend(new_additions)
    for line in additions:
        print(line)
    if problems:
        for line in problems:
            print(f"ERROR: {line}")
        raise SystemExit("Schema compatibility check failed")
    print(f"Schema v1 compatibility OK ({len(additions)} additive changes)")


if __name__ == "__main__":
    main()
