#!/usr/bin/env python3
"""Filter out a specific category from TSV file."""

import argparse
from pathlib import Path


def filter_category(input_path: Path, output_path: Path, exclude_category: str) -> None:
    """Remove all rows with the specified category from TSV file.

    Args:
        input_path: Path to input TSV file
        output_path: Path to output TSV file
        exclude_category: Category to exclude (column 2 in TSV)
    """
    count_original = 0
    count_filtered = 0

    with input_path.open("r", encoding="utf-8") as infile, \
         output_path.open("w", encoding="utf-8") as outfile:

        for line in infile:
            count_original += 1
            parts = line.rstrip("\n").split("\t")

            if len(parts) >= 2:
                category = parts[1]
                if category != exclude_category:
                    outfile.write(line)
                    count_filtered += 1

    print(f"Original rows: {count_original}")
    print(f"Filtered rows: {count_filtered}")
    print(f"Removed rows: {count_original - count_filtered}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter out a specific category from TSV file."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to input TSV file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to output TSV file",
    )
    parser.add_argument(
        "--exclude-category",
        type=str,
        required=True,
        help="Category to exclude (e.g., '1' or '2')",
    )

    args = parser.parse_args()
    filter_category(args.input, args.output, args.exclude_category)


if __name__ == "__main__":
    main()
