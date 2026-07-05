from services.f1_paper_search import SimpleBM25

from evaluation.f1_tuning.f1_evaluation_metrics import (
    mean,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def weighted_cosine(section_scores: dict[str, float], weights: dict[str, float]) -> float:
    total_weight = sum(weights[name] for name in section_scores if name in weights)
    if total_weight == 0:
        return 0.0
    return sum(
        weights[name] * score
        for name, score in section_scores.items()
        if name in weights
    ) / total_weight


def rank_from_query_data(query_data: dict, config: dict, top_k_final: int) -> list[dict]:
    bm25_scores = SimpleBM25(
        k1=config["bm25_k1"],
        b=config["bm25_b"],
    ).score_all(query_data["keywords"], query_data["corpus_for_bm25"])

    results = []
    for paper_id in query_data["candidate_ids"]:
        cosine_score = weighted_cosine(
            query_data["section_scores"].get(paper_id, {}),
            config["section_weights"],
        )
        bm25_score = bm25_scores.get(paper_id, 0.0)
        paper = query_data["candidate_pool"][paper_id].copy()
        paper["paper_id"] = paper_id
        paper["cosine_score"] = round(cosine_score, 4)
        paper["bm25_score"] = round(bm25_score, 4)
        paper["final_score"] = round(
            config["alpha_cosine"] * cosine_score
            + config["alpha_bm25"] * bm25_score,
            4,
        )
        results.append(paper)

    results.sort(key=lambda row: row["final_score"], reverse=True)
    return results[:top_k_final]


def evaluate_config(
    query_data_by_id: dict[str, dict],
    benchmark: list[dict],
    config: dict,
    k: int,
) -> tuple[dict, list[dict]]:
    per_query = []
    for item in benchmark:
        results = rank_from_query_data(
            query_data_by_id[item["id"]],
            config,
            top_k_final=max(k, 10),
        )
        ranked_ids = [str(row.get("paper_id", "")) for row in results]
        top_results = [
            {
                "rank": rank,
                "paper_id": str(row.get("paper_id", "")),
                "title": row.get("title", ""),
                "final_score": row.get("final_score", 0),
                "cosine_score": row.get("cosine_score", 0),
                "bm25_score": row.get("bm25_score", 0),
                "judgment": item["judgments"].get(str(row.get("paper_id", "")), 0),
            }
            for rank, row in enumerate(results[:k], start=1)
        ]
        per_query.append({
            "config": config["name"],
            "query_id": item["id"],
            "query": item["query"],
            f"precision@{k}": precision_at_k(ranked_ids, item["judgments"], k),
            f"recall@{k}": recall_at_k(ranked_ids, item["judgments"], k),
            f"ndcg@{k}": ndcg_at_k(ranked_ids, item["judgments"], k),
            "mrr": reciprocal_rank(ranked_ids, item["judgments"]),
            "top_results": top_results,
        })

    summary = {
        "config": config["name"],
        "alpha_cosine": config["alpha_cosine"],
        "alpha_bm25": config["alpha_bm25"],
        "section_weights": config["section_weights"],
        "bm25_k1": config["bm25_k1"],
        "bm25_b": config["bm25_b"],
        f"precision@{k}": mean([row[f"precision@{k}"] for row in per_query]),
        f"recall@{k}": mean([row[f"recall@{k}"] for row in per_query]),
        f"ndcg@{k}": mean([row[f"ndcg@{k}"] for row in per_query]),
        "mrr": mean([row["mrr"] for row in per_query]),
    }
    return summary, per_query
