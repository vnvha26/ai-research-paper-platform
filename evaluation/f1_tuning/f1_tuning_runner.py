import time
from pathlib import Path

from evaluation.f1_tuning.f1_benchmark_loader import (
    get_benchmark_connections,
    load_benchmark,
    log,
)
from evaluation.f1_tuning.f1_candidate_preparation import prepare_query_data
from evaluation.f1_tuning.f1_config_updater import apply_best_config
from evaluation.f1_tuning.f1_ranking_evaluator import evaluate_config
from evaluation.f1_tuning.f1_tuning_config import CONFIGS, RESULTS_DIR
from evaluation.f1_tuning.f1_tuning_report import (
    print_table,
    write_detail_json,
    write_review_csv,
    write_summary_csv,
)


def _prepare_all_queries(connections, benchmark, top_k_lexical, top_k_semantic):
    query_data_by_id = {}
    for index, item in enumerate(benchmark, start=1):
        query_data_by_id[item["id"]] = prepare_query_data(
            connections=connections,
            item=item,
            query_index=index,
            total_queries=len(benchmark),
            top_k_lexical=top_k_lexical,
            top_k_semantic=top_k_semantic,
        )
    return query_data_by_id


def _evaluate_all_configs(query_data_by_id, benchmark, k):
    summaries = []
    details = []
    log("\n[xếp hạng] Đang đánh giá các cấu hình")
    for index, config in enumerate(CONFIGS, start=1):
        log(f"  - Cấu hình {index}/{len(CONFIGS)}: {config['name']}")
        summary, per_query = evaluate_config(query_data_by_id, benchmark, config, k)
        summaries.append(summary)
        details.extend(per_query)
    summaries.sort(
        key=lambda row: (row[f"ndcg@{k}"], row["mrr"], row[f"precision@{k}"]),
        reverse=True,
    )
    best_config = next(
        config for config in CONFIGS
        if config["name"] == summaries[0]["config"]
    )
    return summaries, details, best_config


def run_tuning(
    benchmark_path: Path,
    config_file: Path,
    k: int = 10,
    top_k_lexical: int = 500,
    top_k_semantic: int = 300,
    review_top_n: int = 50,
    skip_review: bool = False,
) -> int:
    benchmark = load_benchmark(benchmark_path)
    connections = get_benchmark_connections()
    if connections is None:
        log("Không thể kết nối cơ sở dữ liệu. Hãy khởi động Docker và nạp dữ liệu.")
        return 1

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    started = time.time()
    log(f"[chạy] truy vấn={len(benchmark)} cấu hình={len(CONFIGS)} k={k}")
    log(f"[chạy] top_k_từ_khóa={top_k_lexical} top_k_ngữ_nghĩa={top_k_semantic}")
    query_data_by_id = _prepare_all_queries(
        connections,
        benchmark,
        top_k_lexical,
        top_k_semantic,
    )
    summaries, details, best_config = _evaluate_all_configs(
        query_data_by_id,
        benchmark,
        k,
    )

    summary_csv = RESULTS_DIR / f"f1_tuning_summary_at_{k}.csv"
    detail_json = RESULTS_DIR / f"f1_tuning_details_at_{k}.json"
    review_csv = RESULTS_DIR / f"f1_label_review_top_{review_top_n}.csv"
    write_summary_csv(summary_csv, summaries, k)
    write_detail_json(detail_json, details)
    if not skip_review:
        write_review_csv(
            review_csv,
            benchmark,
            query_data_by_id,
            best_config,
            top_n=review_top_n,
        )

    apply_best_config(
        config_file=config_file,
        best_config=best_config,
        top_k_lexical=top_k_lexical,
        top_k_semantic=top_k_semantic,
    )
    print_table(summaries, k)
    log(f"\nĐã lưu tổng hợp: {summary_csv}")
    log(f"Đã lưu chi tiết: {detail_json}")
    if not skip_review:
        log(f"Đã lưu bảng duyệt nhãn: {review_csv}")
        log("Mở CSV để kiểm tra và cập nhật nhãn.")
    log(f"Cấu hình tốt nhất: {summaries[0]['config']}")
    log(f"Tổng thời gian: {time.time() - started:.1f}s")
    return 0
