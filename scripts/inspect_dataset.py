from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("file", type=Path)
    p.add_argument("--n", type=int, default=3)
    args = p.parse_args()

    counts = Counter()
    rows = []
    with args.file.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            row = json.loads(line)
            counts[row["label_name"]] += 1
            if len(rows) < args.n:
                rows.append(row)

    print("label counts:", dict(counts))
    for row in rows:
        print("\n---")
        print(json.dumps(row, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
