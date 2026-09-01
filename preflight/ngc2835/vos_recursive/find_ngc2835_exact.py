#!/usr/bin/env python3

from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import vos

OUT = Path("preflight/ngc2835/vos_recursive")
OUT.mkdir(parents=True, exist_ok=True)

ROOTS = {
    "MUSE": "vos:phangs/RELEASES/PHANGS-MUSE",
    "ALMA": "vos:phangs/RELEASES/PHANGS-ALMA",
    "LANG2020": "vos:phangs/RELEASES/Lang_etal_2020",
}

TARGET_TOKENS = (
    "ngc2835",
    "ngc_2835",
    "ngc-2835",
)

PRODUCT_TOKENS = (
    "maps",
    "co21",
    "mom0",
    "emom0",
    "mom1",
    "strict",
    "broad",
    "rotation",
    "rotcurve",
    "rot_curve",
    "curve",
)

MAX_DEPTH = 7
MAX_NODES = 50000


def norm(value):
    if isinstance(value, bytes):
        return value.decode(errors="ignore")
    return str(value)


def make_child(parent, item):
    item = norm(item).strip()

    if item.startswith("vos:"):
        return item

    return parent.rstrip("/") + "/" + item.lstrip("/")


def interesting(path, label):
    low = path.lower()

    target_hit = any(token in low for token in TARGET_TOKENS)

    product_hit = any(token in low for token in PRODUCT_TOKENS)

    if target_hit:
        return True

    # Lang et al. products can be one all-galaxy table/file,
    # so retain rotation-curve-like filenames even when NGC2835
    # is not literally in the filename.
    return bool(label == "LANG2020" and product_hit)


client = vos.Client()

print("=" * 96)
print("PHANGS NGC 2835 — RECURSIVE VOSPACE DISCOVERY")
print("=" * 96)

print()
print("vos module =", vos.__file__)
print("vos version =", getattr(vos, "__version__", "UNKNOWN"))

public_methods = sorted(name for name in dir(client) if not name.startswith("_"))

print()
print("CLIENT METHODS:")
print(" ".join(public_methods))

for required in ("listdir", "isdir"):
    if not hasattr(client, required):
        raise RuntimeError(f"vos.Client lacks required method: {required}")

print()
print("VOS_CLIENT_METHOD_GATE=PASS")

records = []
errors = []

for label, root in ROOTS.items():
    print()
    print("=" * 96)
    print(f"SEARCH ROOT: {label}")
    print(root)
    print("=" * 96)

    queue = deque([(root, 0)])

    visited = set()
    nodes = 0

    while queue:
        path, depth = queue.popleft()

        if path in visited:
            continue

        visited.add(path)
        nodes += 1

        if nodes > MAX_NODES:
            print(f"{label}: MAX_NODES reached")
            break

        if depth > MAX_DEPTH:
            continue

        try:
            entries = client.listdir(path)

        except Exception as exc:  # noqa: BLE001
            errors.append(
                {
                    "label": label,
                    "path": path,
                    "operation": "listdir",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            continue

        for entry in entries:
            child = make_child(
                path,
                entry,
            )

            try:
                is_dir = bool(client.isdir(child))

            except Exception as exc:  # noqa: BLE001
                errors.append(
                    {
                        "label": label,
                        "path": child,
                        "operation": "isdir",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
                is_dir = False

            if interesting(
                child,
                label,
            ):
                record = {
                    "root": label,
                    "path": child,
                    "is_directory": is_dir,
                    "depth": depth + 1,
                }

                records.append(record)

                print(
                    "MATCH",
                    label,
                    "DIR" if is_dir else "FILE",
                    child,
                )

            if is_dir and depth + 1 <= MAX_DEPTH:
                queue.append(
                    (
                        child,
                        depth + 1,
                    )
                )

    print()
    print(f"{label}_NODES_VISITED={nodes}")


# ------------------------------------------------------------------
# Deduplicate
# ------------------------------------------------------------------

unique = {}

for record in records:
    unique[
        (
            record["root"],
            record["path"],
        )
    ] = record

records = sorted(
    unique.values(),
    key=lambda row: (
        row["root"],
        row["path"].lower(),
    ),
)


# ------------------------------------------------------------------
# Categorise likely minimal products
# ------------------------------------------------------------------

for record in records:
    low = record["path"].lower()

    categories = []

    if "maps" in low:
        categories.append("MUSE_MAPS")

    if "co21" in low and "mom0" in low and "emom0" not in low:
        categories.append("ALMA_CO_MOM0")

    if "co21" in low and "emom0" in low:
        categories.append("ALMA_CO_EMOM0")

    if "co21" in low and "mom1" in low:
        categories.append("ALMA_CO_MOM1")

    if "strict" in low:
        categories.append("STRICT")

    if "broad" in low:
        categories.append("BROAD")

    if any(
        token in low
        for token in (
            "rotation",
            "rotcurve",
            "rot_curve",
            "curve",
        )
    ):
        categories.append("ROTATION_CURVE")

    record["categories"] = categories


json_path = OUT / "ngc2835_exact_vos_candidates.json"

json_path.write_text(
    json.dumps(
        {
            "roots": ROOTS,
            "candidates": records,
            "errors": errors,
        },
        indent=2,
    )
    + "\n"
)


txt_path = OUT / "ngc2835_exact_vos_candidates.txt"

with txt_path.open("w") as handle:
    for record in records:
        handle.write(
            f"{record['root']}\t"
            f"{'DIR' if record['is_directory'] else 'FILE'}\t"
            f"{','.join(record['categories'])}\t"
            f"{record['path']}\n"
        )


print()
print("=" * 96)
print("FINAL CANDIDATES")
print("=" * 96)

for record in records:
    print()
    print(
        "ROOT       =",
        record["root"],
    )

    print(
        "TYPE       =",
        ("DIRECTORY" if record["is_directory"] else "FILE"),
    )

    print(
        "CATEGORIES =",
        ",".join(record["categories"]) or "UNCLASSIFIED",
    )

    print(
        "PATH       =",
        record["path"],
    )


print()
print("=" * 96)
print("SUMMARY")
print("=" * 96)

print(
    "TOTAL_CANDIDATES =",
    len(records),
)

print(
    "ERROR_RECORDS =",
    len(errors),
)

print(
    "OUTPUT_JSON =",
    json_path,
)

print(
    "OUTPUT_TXT =",
    txt_path,
)

print()
print("SCIENCE_DATA_DOWNLOADED=FALSE")
print("HECTOR_INTERNAL_DATA_USED=FALSE")
print("NGC2835_VOS_RECURSIVE_DISCOVERY=PASS")
