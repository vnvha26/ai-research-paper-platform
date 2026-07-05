import math


def precision_at_k(results: list[str], judgments: dict[str, int], k: int) -> float:
    if k <= 0:
        return 0.0
    hits = sum(1 for paper_id in results[:k] if judgments.get(paper_id, 0) > 0)
    return hits / k


def recall_at_k(results: list[str], judgments: dict[str, int], k: int) -> float:
    relevant_count = sum(1 for grade in judgments.values() if grade > 0)
    if relevant_count == 0:
        return 0.0
    hits = sum(1 for paper_id in results[:k] if judgments.get(paper_id, 0) > 0)
    return hits / relevant_count


def dcg_at_k(grades: list[int], k: int) -> float:
    score = 0.0
    for index, grade in enumerate(grades[:k], start=1):
        score += ((2 ** grade) - 1) / math.log2(index + 1)
    return score


def ndcg_at_k(results: list[str], judgments: dict[str, int], k: int) -> float:
    observed = [judgments.get(paper_id, 0) for paper_id in results[:k]]
    ideal = dcg_at_k(sorted(judgments.values(), reverse=True), k)
    return dcg_at_k(observed, k) / ideal if ideal else 0.0


def reciprocal_rank(results: list[str], judgments: dict[str, int]) -> float:
    for index, paper_id in enumerate(results, start=1):
        if judgments.get(paper_id, 0) > 0:
            return 1 / index
    return 0.0


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
