#!/usr/bin/env python3
"""
Custom ensemble majority vote with different threshold for Kompatibles_Fahrzeug_Modell.

For most tags: requires min_votes
For Kompatibles_Fahrzeug_Modell: requires min_votes + k
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
    kfm_boost: int
) -> List[Record]:
    """
    Perform majority vote with custom threshold for Kompatibles_Fahrzeug_Modell.
    
    Args:
        records: Iterable of record lists from each submission
        min_votes: Minimum votes required for most tags
        kfm_boost: Additional votes required for Kompatibles_Fahrzeug_Modell
    
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
        
        if aspect == "Kompatibles_Fahrzeug_Modell":
            # Higher threshold for Kompatibles_Fahrzeug_Modell
            if count >= (min_votes + kfm_boost):
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
        description='Custom ensemble majority vote with different threshold for Kompatibles_Fahrzeug_Modell'
    )
    parser.add_argument('--inputs', nargs='+', required=True, help='Input TSV files')
    parser.add_argument('--output', required=True, help='Output TSV file')
    parser.add_argument('--votes', type=int, required=True, help='Minimum votes for most tags')
    parser.add_argument('--kfm-boost', type=int, default=0, 
                        help='Additional votes required for Kompatibles_Fahrzeug_Modell (default: 0)')
    
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
    print(f"Minimum votes for Kompatibles_Fahrzeug_Modell: {args.votes + args.kfm_boost}")
    
    # Perform custom majority vote
    ensembled = custom_majority_vote(all_records, args.votes, args.kfm_boost)
    
    # Count records by aspect type
    kfm_count = sum(1 for r in ensembled if r[2] == "Kompatibles_Fahrzeug_Modell")
    other_count = len(ensembled) - kfm_count
    
    # Write output
    output_path = Path(args.output)
    with output_path.open('w', encoding='utf-8') as f:
        for record_id, category, aspect, value in ensembled:
            f.write(f'{record_id}\t{category}\t{aspect}\t{value}\n')
    
    print(f"\nResults:")
    print(f"  Kompatibles_Fahrzeug_Modell: {kfm_count} records")
    print(f"  Other tags: {other_count} records")
    print(f"  Total: {len(ensembled)} records")
    print(f"\nWrote {len(ensembled)} ensembled rows to {args.output}")

if __name__ == '__main__':
    main()
