#!/usr/bin/env python3
"""
Custom ensemble majority vote with different threshold for a specific tag.

For most tags: requires min_votes
For specified tag: requires min_votes + boost
"""

import argparse
from collections import Counter
from pathlib import Path
from typing import List, Tuple, Iterable

Record = Tuple[int, str, str, str]

def read_tsv(path: Path) -> List[Record]:
    """Read TSV file and return list of records."""
    records = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 4:
                record_id = int(parts[0])
                category = parts[1]
                aspect = parts[2]
                value = parts[3]
                records.append((record_id, category, aspect, value))
    return records

def custom_majority_vote(
    records: Iterable[List[Record]], 
    min_votes: int,
    boost: int,
    target_tag: str
) -> List[Record]:
    """
    Perform majority vote with custom threshold for specific tag.
    
    Args:
        records: Iterable of record lists from each submission
        min_votes: Minimum votes required for most tags
        boost: Additional votes required for target tag
        target_tag: The tag to apply boost to
    
    Returns:
        List of records that meet the threshold
    """
    counter: Counter[Record] = Counter()
    
    # Count occurrences of each prediction
    for record_list in records:
        counter.update(record_list)
    
    # Apply different thresholds based on aspect
    majority_records = []
    for record, count in counter.items():
        aspect = record[2]  # aspect is at index 2
        
        if aspect == target_tag:
            # Custom threshold for target tag
            if count >= (min_votes + boost):
                majority_records.append(record)
        else:
            # Regular threshold for other tags
            if count >= min_votes:
                majority_records.append(record)
    
    # Sort by record_id, category, aspect, value
    majority_records.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    
    return majority_records

def main():
    parser = argparse.ArgumentParser(
        description='Custom ensemble majority vote with different threshold for specific tag'
    )
    parser.add_argument('--inputs', nargs='+', required=True, help='Input TSV files')
    parser.add_argument('--output', required=True, help='Output TSV file')
    parser.add_argument('--votes', type=int, required=True, help='Minimum votes for most tags')
    parser.add_argument('--boost', type=int, default=0, 
                        help='Additional votes required for target tag (default: 0)')
    parser.add_argument('--target-tag', required=True,
                        help='Tag to apply boost to (e.g., Kompatible_Fahrzeug_Marke)')
    
    args = parser.parse_args()
    
    # Read all input files
    all_records = []
    for input_file in args.inputs:
        path = Path(input_file)
        if not path.exists():
            print(f"Warning: {input_file} does not exist, skipping")
            continue
        records = read_tsv(path)
        all_records.append(records)
    
    print(f"Read {len(all_records)} submission files")
    print(f"Minimum votes for regular tags: {args.votes}")
    print(f"Minimum votes for {args.target_tag}: {args.votes + args.boost}")
    
    # Perform custom majority vote
    ensembled = custom_majority_vote(all_records, args.votes, args.boost, args.target_tag)
    
    # Count records by aspect type
    target_count = sum(1 for r in ensembled if r[2] == args.target_tag)
    other_count = len(ensembled) - target_count
    
    # Write output
    output_path = Path(args.output)
    with output_path.open('w', encoding='utf-8') as f:
        for record_id, category, aspect, value in ensembled:
            f.write(f'{record_id}\t{category}\t{aspect}\t{value}\n')
    
    print(f"\nResults:")
    print(f"  {args.target_tag}: {target_count} records")
    print(f"  Other tags: {other_count} records")
    print(f"  Total: {len(ensembled)} records")
    print(f"\nWrote {len(ensembled)} ensembled rows to {args.output}")

if __name__ == '__main__':
    main()
