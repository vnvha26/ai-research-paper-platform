from services.f2_citation_fraud.f2_fraud_config import (
    COMMON_NEIGHBOR_THRESHOLD,
    LOCAL_DENSITY_MIN_JACCARD,
    LOCAL_DENSITY_THRESHOLD,
    LOW_RELEVANCE_THRESHOLD,
    NEIGHBOR_JACCARD_THRESHOLD,
)
from services.f2_citation_fraud.f2_content_analysis import (
    build_bm25_scores,
    encode_sections,
    normalize_author_ids,
    weighted_section_cosine,
)
from services.f2_citation_fraud.f2_citation_graph import fetch_citation_records
from services.f2_citation_fraud.f2_paper_repository import fetch_papers
from services.f2_citation_fraud.f2_risk_assessment import graph_metrics, status_from_signals
from services.f1_paper_search.f1_search_config import ALPHA_BM25, ALPHA_COSINE


def _author_overlap(source_author_ids, source_author_names, cited_authors):
    cited_author_ids, cited_author_names = normalize_author_ids(cited_authors)
    shared_ids = sorted(source_author_ids.intersection(cited_author_ids))
    shared_names = [
        source_author_names.get(author_id)
        or cited_author_names.get(author_id)
        or author_id
        for author_id in shared_ids
    ]
    union_count = len(source_author_ids.union(cited_author_ids))
    overlap_ratio = len(shared_ids) / union_count if union_count else 0.0
    return shared_ids, shared_names, overlap_ratio


def _content_scores(nlp_model, source_vectors, cited_data, bm25_scores, cited_id):
    cited_vectors = encode_sections(nlp_model, cited_data)
    section_cosine, section_scores = weighted_section_cosine(source_vectors, cited_vectors)
    bm25_score = bm25_scores.get(cited_id, 0.0)
    content_score = section_cosine
    if bm25_scores:
        content_score = ALPHA_COSINE * section_cosine + ALPHA_BM25 * bm25_score
    return content_score, section_cosine, bm25_score, section_scores


def _analyze_record(
    record,
    cited_data,
    nlp_model,
    source_vectors,
    source_author_ids,
    source_author_names,
    bm25_scores,
):
    cited_id = str(record["cited_id"])
    content_score, section_cosine, bm25_score, section_scores = _content_scores(
        nlp_model,
        source_vectors,
        cited_data,
        bm25_scores,
        cited_id,
    )
    metrics = graph_metrics(record)
    shared_ids, shared_names, author_overlap_ratio = _author_overlap(
        source_author_ids,
        source_author_names,
        cited_data.get("authors"),
    )
    low_relevance = content_score < LOW_RELEVANCE_THRESHOLD
    is_mutual = bool(record["is_mutual"])
    risk_level, status = status_from_signals(
        low_relevance,
        is_mutual,
        bool(shared_ids),
        metrics["graph_abnormal"],
    )

    return {
        "id": cited_id,
        "title": record.get("cited_title") or cited_data.get("title") or "Không có tiêu đề",
        "similarity": float(content_score),
        "content_score": float(content_score),
        "section_cosine": float(section_cosine),
        "bm25_score": float(bm25_score),
        "section_scores": section_scores,
        "is_mutual": is_mutual,
        "shared_author_count": len(shared_ids),
        "shared_authors": shared_names,
        "author_overlap_ratio": float(author_overlap_ratio),
        "graph_abnormal": metrics["graph_abnormal"],
        "neighbor_jaccard": float(metrics["neighbor_jaccard"]),
        "common_neighbor_count": metrics["common_neighbor_count"],
        "union_neighbor_count": metrics["union_neighbor_count"],
        "source_neighbor_count": int(record.get("source_neighbor_count") or 0),
        "target_neighbor_count": int(record.get("target_neighbor_count") or 0),
        "common_neighbor_sample": record.get("common_neighbor_sample") or [],
        "local_density": float(metrics["local_density"]),
        "local_node_count": metrics["local_node_count"],
        "local_edge_count": metrics["local_edge_count"],
        "low_relevance": low_relevance,
        "status": status,
        "risk_level": risk_level,
    }


def _build_result(details, graph_citation_count, skipped_ids):
    red_flags = sum(item["risk_level"] == 2 for item in details)
    yellow_flags = sum(item["risk_level"] == 1 for item in details)
    total_citations = len(details)
    fraud_score = 0.0
    if total_citations:
        fraud_score = ((red_flags * 2 + yellow_flags) / (total_citations * 2)) * 100

    sorted_details = sorted(
        details,
        key=lambda item: (
            item["risk_level"],
            item["graph_abnormal"],
            item["shared_author_count"],
            item["neighbor_jaccard"],
            -item["content_score"],
        ),
        reverse=True,
    )
    return {
        "total_citations": total_citations,
        "graph_citation_count": graph_citation_count,
        "skipped_missing_dataset": len(skipped_ids),
        "skipped_missing_ids": skipped_ids[:20],
        "fraud_score": min(fraud_score, 100.0),
        "red_flags": red_flags,
        "yellow_flags": yellow_flags,
        "content_threshold": LOW_RELEVANCE_THRESHOLD,
        "neighbor_jaccard_threshold": NEIGHBOR_JACCARD_THRESHOLD,
        "common_neighbor_threshold": COMMON_NEIGHBOR_THRESHOLD,
        "local_density_threshold": LOCAL_DENSITY_THRESHOLD,
        "local_density_min_jaccard": LOCAL_DENSITY_MIN_JACCARD,
        "details": sorted_details,
    }


def analyze_paper_fraud(connections, paper_id):
    try:
        records = fetch_citation_records(connections["neo4j"], paper_id)
    except Exception as exc:
        return {"error": f"Lỗi Graph DB: {exc}"}

    if not records or not records[0].get("cited_id"):
        return {"error": "Bài báo này không trích dẫn bài nào trong tập dữ liệu của hệ thống."}

    cited_ids = [str(record["cited_id"]) for record in records if record.get("cited_id")]
    graph_citation_count = len(cited_ids)
    try:
        papers = fetch_papers(connections["pg"], [str(paper_id)] + cited_ids)
    except RuntimeError as exc:
        return {"error": str(exc)}
    source_data = papers.get(str(paper_id))
    if not source_data:
        return {"error": "Không tìm thấy nội dung bài báo gốc trong PostgreSQL."}

    try:
        source_vectors = encode_sections(connections["nlp_model"], source_data)
    except Exception as exc:
        return {"error": f"Lỗi tạo embedding: {exc}"}
    if not source_vectors:
        return {"error": "Bài báo gốc không có đủ text để tính embedding."}

    source_author_ids, source_author_names = normalize_author_ids(source_data.get("authors"))
    cited_rows = {paper_id: papers[paper_id] for paper_id in cited_ids if paper_id in papers}
    skipped_ids = [paper_id for paper_id in cited_ids if paper_id not in cited_rows]
    if not cited_rows:
        return {
            "error": (
                "Bài báo này có citation edge trong graph, nhưng không có paper được cite nào "
                "tồn tại trong bảng papers của dataset nên F2 không có dữ liệu để phân tích."
            ),
            "graph_citation_count": graph_citation_count,
            "skipped_missing_dataset": len(skipped_ids),
        }

    bm25_scores = build_bm25_scores(source_data, cited_rows)
    details = []
    for record in records:
        if not record.get("cited_id"):
            continue
        cited_id = str(record["cited_id"])
        cited_data = cited_rows.get(cited_id)
        if cited_data:
            try:
                details.append(_analyze_record(
                    record,
                    cited_data,
                    connections["nlp_model"],
                    source_vectors,
                    source_author_ids,
                    source_author_names,
                    bm25_scores,
                ))
            except Exception as exc:
                return {"error": f"Lỗi phân tích trích dẫn {cited_id}: {exc}"}
    return _build_result(details, graph_citation_count, skipped_ids)
