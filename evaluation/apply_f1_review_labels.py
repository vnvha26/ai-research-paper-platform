import argparse
import csv
import json
from pathlib import Path


BENCHMARK_FILE = Path(__file__).with_name("f1_benchmark_queries.json")
REVIEW_FILE = Path(__file__).with_name("results") / "f1_label_review_top_50.csv"


def load_benchmark(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        benchmark = json.load(f)
    for item in benchmark:
        item["judgments"] = {str(k): int(v) for k, v in item.get("judgments", {}).items()}
    return benchmark


def apply_review(benchmark: list[dict], review_path: Path, use_current_if_empty: bool = False) -> int:
    by_query_id = {item["id"]: item for item in benchmark}
    updates = 0

    with review_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            query_id = (row.get("query_id") or "").strip()
            paper_id = (row.get("paper_id") or "").strip()
            new_value = (row.get("new_judgment_0_3") or "").strip()
            if use_current_if_empty and new_value == "":
                new_value = (row.get("current_judgment") or "").strip()

            if not query_id or not paper_id or new_value == "":
                continue
            if query_id not in by_query_id:
                continue

            try:
                grade = int(new_value)
            except ValueError:
                continue
            if grade < 0 or grade > 3:
                continue

            judgments = by_query_id[query_id].setdefault("judgments", {})
            if grade == 0:
                judgments.pop(paper_id, None)
            else:
                judgments[paper_id] = grade
            updates += 1

    return updates


def save_benchmark(path: Path, benchmark: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(benchmark, f, ensure_ascii=False, indent=2)
        f.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply manual F1 relevance labels from review CSV.")
    parser.add_argument("--benchmark", type=Path, default=BENCHMARK_FILE)
    parser.add_argument("--review", type=Path, default=REVIEW_FILE)
    parser.add_argument(
        "--use-current-if-empty",
        action="store_true",
        help="Use current_judgment when new_judgment_0_3 is empty.",
    )
    args = parser.parse_args()

    if not args.review.exists():
        print(f"Review file not found: {args.review}")
        return 1

    benchmark = load_benchmark(args.benchmark)
    updates = apply_review(
        benchmark,
        args.review,
        use_current_if_empty=args.use_current_if_empty,
    )
    save_benchmark(args.benchmark, benchmark)

    print(f"Applied {updates} reviewed labels to {args.benchmark}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
