import csv
import json
from pathlib import Path

from evaluation.f1_tuning.f1_benchmark_loader import log
from evaluation.f1_tuning.f1_ranking_evaluator import rank_from_query_data


def snippet(text: str, max_len: int = 260) -> str:
    text = " ".join((text or "").split())
    if len(text) <= max_len:
        return text
    return text[:max_len - 3] + "..."


def write_review_csv(
    path: Path,
    benchmark: list[dict],
    query_data_by_id: dict[str, dict],
    best_config: dict,
    top_n: int,
) -> None:
    fieldnames = [
        "query_id", "query", "rank", "paper_id", "current_judgment",
        "new_judgment_0_3", "title", "abstract", "intro", "method", "conclusion",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for item in benchmark:
            query_data = query_data_by_id[item["id"]]
            results = rank_from_query_data(query_data, best_config, top_k_final=top_n)
            for rank, row in enumerate(results, start=1):
                paper_id = str(row["paper_id"])
                sections = query_data["sections_map"].get(paper_id, {})
                writer.writerow({
                    "query_id": item["id"],
                    "query": item["query"],
                    "rank": rank,
                    "paper_id": paper_id,
                    "current_judgment": item["judgments"].get(paper_id, 0),
                    "new_judgment_0_3": "",
                    "title": row.get("title", ""),
                    "abstract": snippet(row.get("abstract", "") or sections.get("abstract", "")),
                    "intro": snippet(sections.get("intro", "")),
                    "method": snippet(sections.get("method", "")),
                    "conclusion": snippet(sections.get("conclusion", "")),
                })


def write_summary_csv(path: Path, rows: list[dict], k: int) -> None:
    fieldnames = [
        "config", "alpha_cosine", "alpha_bm25", "section_weights",
        "bm25_k1", "bm25_b", f"precision@{k}", f"recall@{k}",
        f"ndcg@{k}", "mrr",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            csv_row = row.copy()
            csv_row["section_weights"] = json.dumps(
                csv_row["section_weights"],
                ensure_ascii=False,
            )
            writer.writerow(csv_row)


def write_detail_json(path: Path, details: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(details, file, ensure_ascii=False, indent=2)


def print_table(rows: list[dict], k: int) -> None:
    metric_names = [f"precision@{k}", f"recall@{k}", f"ndcg@{k}", "mrr"]
    log("\nKết quả tinh chỉnh F1")
    log("-" * 92)
    log(f"{'cấu hình':32} {'P@'+str(k):>8} {'R@'+str(k):>8} {'NDCG@'+str(k):>10} {'MRR':>8}")
    log("-" * 92)
    for row in rows:
        log(
            f"{row['config'][:32]:32} "
            f"{row[metric_names[0]]:8.4f} "
            f"{row[metric_names[1]]:8.4f} "
            f"{row[metric_names[2]]:10.4f} "
            f"{row[metric_names[3]]:8.4f}"
        )
