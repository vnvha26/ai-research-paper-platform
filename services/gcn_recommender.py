import logging

try:
    from qdrant_client.models import Filter, FieldCondition, MatchValue
except ModuleNotFoundError:
    Filter = None
    FieldCondition = None
    MatchValue = None


logger = logging.getLogger(__name__)


def recommend_related_papers(connections, paper_id, top_k=5):
    """
    Recommend papers related to the current paper by using GCN embeddings.

    The current paper is first looked up in Qdrant gcn_collection. Its GCN
    vector is then used to search for nearest neighbor papers in the same
    collection.
    """
    if not connections or "qdrant" not in connections:
        return []

    if Filter is None or FieldCondition is None or MatchValue is None:
        return []

    qdrant = connections["qdrant"]
    paper_id = str(paper_id)

    source_filter = Filter(
        must=[
            FieldCondition(
                key="paper_id",
                match=MatchValue(value=paper_id)
            )
        ]
    )

    try:
        scroll_result = qdrant.scroll(
            collection_name="gcn_collection",
            scroll_filter=source_filter,
            limit=1,
            with_payload=True,
            with_vectors=True
        )
    except Exception as exc:
        logger.exception("Không thể lấy vector GCN của bài %s: %s", paper_id, exc)
        return []

    source_points = scroll_result[0] if isinstance(scroll_result, tuple) else scroll_result
    if not source_points:
        return []

    source_vector = source_points[0].vector
    if not source_vector:
        return []

    try:
        if hasattr(qdrant, "query_points"):
            hits = qdrant.query_points(
                collection_name="gcn_collection",
                query=source_vector,
                limit=top_k + 5,
                with_payload=True
            ).points
        else:
            hits = qdrant.search(
                collection_name="gcn_collection",
                query_vector=source_vector,
                limit=top_k + 5,
                with_payload=True
            )
    except Exception as exc:
        logger.exception("Không thể tìm bài liên quan cho %s: %s", paper_id, exc)
        return []

    recommendations = []
    seen_ids = {paper_id}

    for hit in hits:
        payload = hit.payload or {}
        hit_pid = str(payload.get("paper_id", ""))

        if not hit_pid or hit_pid in seen_ids:
            continue

        seen_ids.add(hit_pid)
        recommendations.append({
            "paper_id": hit_pid,
            "title": payload.get("title", "Khong co tieu de"),
            "year": payload.get("year", "N/A"),
            "gcn_score": float(hit.score)
        })

        if len(recommendations) >= top_k:
            break

    return recommendations
