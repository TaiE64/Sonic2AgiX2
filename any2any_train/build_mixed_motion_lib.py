"""Build a symlink-only WBT/locomotion rehearsal motion library."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def locomotion_repeat_count(wbt_count: int, locomotion_count: int, share: float) -> int:
    if wbt_count <= 0 or locomotion_count <= 0:
        raise ValueError("both motion libraries must contain at least one pickle")
    if not 0.0 < share < 1.0:
        raise ValueError("locomotion share must be between zero and one")
    return max(1, math.ceil(share * wbt_count / ((1.0 - share) * locomotion_count)))


def _link(target: Path, source: Path):
    if target.exists() or target.is_symlink():
        if target.is_symlink() and target.resolve() == source.resolve():
            return
        raise FileExistsError(f"refusing to replace existing entry: {target}")
    target.symlink_to(source.resolve())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wbt", type=Path, required=True)
    parser.add_argument("--locomotion", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--locomotion-share", type=float, default=0.30)
    args = parser.parse_args()

    wbt_files = sorted(p for p in args.wbt.rglob("*.pkl") if p.name != "metadata.pkl")
    locomotion_files = sorted(
        p for p in args.locomotion.rglob("*.pkl") if p.name != "metadata.pkl"
    )
    repeats = locomotion_repeat_count(
        len(wbt_files), len(locomotion_files), args.locomotion_share
    )
    args.out.mkdir(parents=True, exist_ok=True)

    for index, source in enumerate(wbt_files):
        # The index guarantees uniqueness; truncation avoids filesystem name
        # limits for deeply nested source libraries.
        label = source.stem[:180]
        _link(args.out / f"wbt__{index:05d}__{label}.pkl", source)
    for repeat in range(repeats):
        for index, source in enumerate(locomotion_files):
            _link(
                args.out / f"locomotion__r{repeat:03d}__{index:03d}__{source.name}",
                source,
            )

    locomotion_entries = repeats * len(locomotion_files)
    total = len(wbt_files) + locomotion_entries
    report = {
        "wbt_source": str(args.wbt.resolve()),
        "locomotion_source": str(args.locomotion.resolve()),
        "wbt_entries": len(wbt_files),
        "locomotion_unique": len(locomotion_files),
        "locomotion_repeats": repeats,
        "locomotion_entries": locomotion_entries,
        "locomotion_share": locomotion_entries / total,
        "total_entries": total,
    }
    with (args.out / "mix_manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
