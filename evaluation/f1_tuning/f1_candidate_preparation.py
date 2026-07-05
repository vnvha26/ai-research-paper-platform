import time

from services.f1_paper_search import (
    extract_keywords,
    fetch_sections,
    get_section_scores_batch,
    lexical_search,
)
from services.f1_paper_search.f1_search_config import MULTIVEC_COLLECTION

from evaluation.f1_tuning.f1_benchmark_loader import log


def query_qdrant(connections: dict, query_vector: list[float], limit: int, query_filter=None):
    qdrant = connections["qdrant"]
    if hasattr(qdrant, "query_points"):
        return qdrant.query_points(
            collection_name=MULTIVEC_COLLECTION,
            query=query_vector,
            query_filter=query_filter,
            limit=limit,
        ).points
    return qdrant.search(
        collection_name=MULTIVEC_COLLECTION,
        query_vector=query_vector,
        query_filter=query_filter,
        limit=limit,
    )


def _collect_candidate_pool(connections, keywords, query_vector, top_k_lexical, top_k_semantic):
    candidates = {}
    for paper in lexical_search(connections["pg"], keywords, top_k=top_k_lexical):
        candidates[str(paper["paper_id"])] = paper
    for hit in query_qdrant(connections, query_vector, limit=top_k_semantic):
        paper_id = str(hit.payload.get("paper_id", ""))
        if paper_id and paper_id not in candidates:
            candidates[paper_id] = {
                "paper_id": paper_id,
                "title": hit.payload.get("title", ""),
                "year": hit.payload.get("year", ""),
                "abstract": "",
            }
    return candidates


def _load_section_scores(connections, query_vector, candidate_ids):
    log(f"    cosine batch cho {len(candidate_ids)} ứng viên")
    return get_section_scores_batch(
        connections["qdrant"],
        query_vector,
        candidate_ids,
    )


def _build_bm25_corpus(candidate_ids, sections_map):
    return [
        {
            "paper_id": paper_id,
            "abstract": sections_map.get(paper_id, {}).get("abstract", ""),
            "intro": sections_map.get(paper_id, {}).get("intro", ""),
            "method": sections_map.get(paper_id, {}).get("method", ""),
            "conclusion": sections_map.get(paper_id, {}).get("conclusion", ""),
        }
        for paper_id in candidate_ids
    ]


def prepare_query_data(
    connections: dict,
    item: dict,
    query_index: int,
    total_queries: int,
    top_k_lexical: int,
    top_k_semantic: int,
) -> dict:
    started = time.time()
    query = item["query"]
    log(f"\n[truy vấn {query_index}/{total_queries}] {query}")
    log("  - Mã hóa truy vấn")
    query_vector = connections["nlp_model"].encode([query])[0].tolist()
    keywords = extract_keywords(query)

    log(f"  - Ứng viên PSQL (top_k={top_k_lexical})")
    log(f"  - Ứng viên Qdrant (top_k={top_k_semantic})")
    candidate_pool = _collect_candidate_pool(
        connections,
        keywords,
        query_vector,
        top_k_lexical,
        top_k_semantic,
    )
    candidate_ids = list(candidate_pool)
    log(f"  - Tổng ứng viên: {len(candidate_ids)}")

    log("  - Lấy nội dung từ PostgreSQL")
    sections_map = fetch_sections(connections["pg"], candidate_ids)
    for paper_id, sections in sections_map.items():
        if paper_id in candidate_pool and not candidate_pool[paper_id].get("abstract"):
            candidate_pool[paper_id]["abstract"] = sections["abstract"]

    log("  - Tính điểm cosine từng phần")
    section_scores = _load_section_scores(connections, query_vector, candidate_ids)
    log(f"  - Hoàn tất sau {time.time() - started:.1f}s")
    return {
        "item": item,
        "keywords": keywords,
        "candidate_ids": candidate_ids,
        "candidate_pool": candidate_pool,
        "sections_map": sections_map,
        "section_scores": section_scores,
        "corpus_for_bm25": _build_bm25_corpus(candidate_ids, sections_map),
    }
