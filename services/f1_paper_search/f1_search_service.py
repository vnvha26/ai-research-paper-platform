try:
    import streamlit as st
except ModuleNotFoundError:
    class _StreamlitFallback:
        @staticmethod
        def error(message):
            print(message)

        @staticmethod
        def warning(message):
            print(message)

    st = _StreamlitFallback()

from services.f1_paper_search.f1_search_config import (
    ALPHA_BM25,
    ALPHA_COSINE,
    BM25_B,
    BM25_K1,
    TOP_K_LEXICAL,
    TOP_K_SEMANTIC,
)
from services.f1_paper_search.f1_query_ranking import SimpleBM25, extract_keywords
from services.f1_paper_search.f1_candidate_retrieval import (
    fetch_sections,
    get_weighted_cosine_scores,
    lexical_search,
    semantic_search,
)


def _collect_candidates(pg_conn, qdrant, keywords, query_vector, lexical_limit, semantic_limit):
    candidates = {
        paper["paper_id"]: paper
        for paper in lexical_search(pg_conn, keywords, top_k=lexical_limit)
    }
    try:
        for hit in semantic_search(qdrant, query_vector, semantic_limit):
            paper_id = hit.payload.get("paper_id")
            if paper_id and paper_id not in candidates:
                candidates[paper_id] = {
                    "paper_id": paper_id,
                    "title": hit.payload.get("title", ""),
                    "year": hit.payload.get("year", ""),
                    "abstract": "",
                }
    except Exception as exc:
        st.error(f"Lỗi Qdrant: {exc}")
    return candidates


def _build_bm25_corpus(candidate_ids, sections):
    return [
        {
            "paper_id": paper_id,
            "abstract": sections.get(paper_id, {}).get("abstract", ""),
            "intro": sections.get(paper_id, {}).get("intro", ""),
            "method": sections.get(paper_id, {}).get("method", ""),
            "conclusion": sections.get(paper_id, {}).get("conclusion", ""),
        }
        for paper_id in candidate_ids
    ]


def _embedding_quality(section_data):
    available = sum(bool(section_data.get(name)) for name in ("intro", "method", "conclusion"))
    if available == 3:
        return "full"
    if available >= 1:
        return "partial"
    return "title_only"


def hybrid_search_f1(
    connections: dict,
    user_query: str,
    top_k_final: int = 30,
    alpha_cosine: float = ALPHA_COSINE,
    alpha_bm25: float = ALPHA_BM25,
    section_weights: dict[str, float] | None = None,
    bm25_k1: float = BM25_K1,
    bm25_b: float = BM25_B,
    top_k_lexical: int = TOP_K_LEXICAL,
    top_k_semantic: int = TOP_K_SEMANTIC,
) -> list[dict]:
    pg_conn = connections["pg"]
    qdrant = connections["qdrant"]
    query_vector = connections["nlp_model"].encode([user_query])[0].tolist()
    keywords = extract_keywords(user_query)

    try:
        candidates = _collect_candidates(
            pg_conn,
            qdrant,
            keywords,
            query_vector,
            top_k_lexical,
            top_k_semantic,
        )
    except RuntimeError as exc:
        st.error(str(exc))
        return []
    if not candidates:
        st.warning("Không tìm thấy bài báo nào. Hãy thử từ khóa khác!")
        return []

    candidate_ids = list(candidates)
    try:
        sections = fetch_sections(pg_conn, candidate_ids)
    except RuntimeError as exc:
        st.error(str(exc))
        return []
    for paper_id, section_data in sections.items():
        if paper_id in candidates and not candidates[paper_id].get("abstract"):
            candidates[paper_id]["abstract"] = section_data["abstract"]

    bm25_scores = SimpleBM25(k1=bm25_k1, b=bm25_b).score_all(
        keywords,
        _build_bm25_corpus(candidate_ids, sections),
    )
    try:
        cosine_scores = get_weighted_cosine_scores(
            qdrant,
            query_vector,
            candidate_ids,
            section_weights=section_weights,
        )
    except RuntimeError as exc:
        st.error(str(exc))
        return []
    results = []
    for paper_id in candidate_ids:
        cosine_score = cosine_scores.get(paper_id, 0.0)
        bm25_score = bm25_scores.get(paper_id, 0.0)
        paper = candidates[paper_id].copy()
        paper["cosine_score"] = round(cosine_score, 4)
        paper["bm25_score"] = round(bm25_score, 4)
        paper["final_score"] = round(
            alpha_cosine * cosine_score + alpha_bm25 * bm25_score,
            4,
        )
        paper["embedding_quality"] = _embedding_quality(sections.get(paper_id, {}))
        results.append(paper)

    results.sort(key=lambda paper: paper["final_score"], reverse=True)
    return results[:top_k_final]
