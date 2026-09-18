from __future__ import annotations

import argparse
import csv
from pathlib import Path


HEADER = ["trial_id", "reason"]


def read_rows(path: Path):
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != HEADER:
            raise RuntimeError(
                f"Expected columns {HEADER}, got {reader.fieldnames}: {path}"
            )
        return [dict(row) for row in reader]


def write_rows(path: Path, rows: list[dict[str, str]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=HEADER)
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Manage SI2026 trials excluded from paper analysis."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("./manual_release_tracking/si2026_data"),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    add_parser = subparsers.add_parser("add")
    add_parser.add_argument("--trial-id", required=True)
    add_parser.add_argument("--reason", required=True)

    remove_parser = subparsers.add_parser("remove")
    remove_parser.add_argument("--trial-id", required=True)

    subparsers.add_parser("list")
    return parser.parse_args()


def main():
    args = parse_args()
    path = args.data_dir / "trial_exclusions.csv"
    rows = read_rows(path)

    if args.command == "list":
        if not rows:
            print("No manually excluded trials.")
            return
        for row in rows:
            print(f"{row['trial_id']}\t{row['reason']}")
        return

    if args.command == "add":
        replacement = {
            "trial_id": args.trial_id.strip(),
            "reason": args.reason.strip(),
        }
        if not replacement["trial_id"] or not replacement["reason"]:
            raise SystemExit("--trial-id and --reason must not be blank")
        rows = [row for row in rows if row["trial_id"] != replacement["trial_id"]]
        rows.append(replacement)
        write_rows(path, rows)
        print(f"Excluded {replacement['trial_id']}: {replacement['reason']}")
        return

    original_count = len(rows)
    rows = [row for row in rows if row["trial_id"] != args.trial_id]
    if len(rows) == original_count:
        raise SystemExit(f"Trial was not in the exclusions file: {args.trial_id}")
    write_rows(path, rows)
    print(f"Removed exclusion: {args.trial_id}")


if __name__ == "__main__":
    main()
