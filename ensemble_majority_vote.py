#!/usr/bin/env python3
"""Majority-vote ensembling for TSV submissions."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Tuple

Record = Tuple[int, str, str, str]


def read_submission(path: Path) -> List[Record]:
    seen: set[Record] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 4:
                raise ValueError(f"Expected 4 tab-separated fields in {path} line {line_no}, got {len(parts)}")
            record_id_str, category, aspect, value = parts
            try:
                record_id = int(record_id_str)
            except ValueError as exc:
                raise ValueError(f"Record id must be an integer in {path} line {line_no}: {record_id_str}") from exc
            entry: Record = (record_id, category, aspect, value)
            seen.add(entry)
    return list(seen)


def majority_vote(records: Iterable[List[Record]], min_votes: int) -> List[Record]:
    counter: Counter[Record] = Counter()
    for record_list in records:
        counter.update(record_list)
    majority_records = [record for record, count in counter.items() if count >= min_votes]
    majority_records.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    return majority_records


def write_submission(records: List[Record], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record_id, category, aspect, value in records:
            handle.write(f"{record_id}\t{category}\t{aspect}\t{value}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ensemble TSV submissions via majority vote.")
    parser.add_argument(
        "--inputs",
        nargs="+",
        type=Path,
        required=True,
        help="Paths to input TSV files (>=2).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to write ensembled TSV.",
    )
    parser.add_argument(
        "--votes",
        type=int,
        default=None,
        help="Minimum votes required to keep a record (default: simple majority).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.inputs) < 2:
        raise ValueError("Need at least two input files for ensembling")

    submissions = [read_submission(path) for path in args.inputs]
    majority = args.votes if args.votes is not None else (len(submissions) // 2) + 1
    ensembled = majority_vote(submissions, majority)
    write_submission(ensembled, args.output)
    print(f"Wrote {len(ensembled)} ensembled rows to {args.output}")


if __name__ == "__main__":
    main()
