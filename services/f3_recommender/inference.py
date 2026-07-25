import logging

try:
    from qdrant_client.models import FieldCondition, Filter, MatchValue
except ModuleNotFoundError:
    FieldCondition = Filter = MatchValue = None


logger = logging.getLogger(__name__)


def recommend_related_papers(connections, paper_id, top_k=5):
    if not connections or "qdrant" not in connections:
        return []
    if Filter is None or FieldCondition is None or MatchValue is None:
        return []

    qdrant = connections["qdrant"]
    paper_id = str(paper_id)
    source_filter = Filter(
        must=[FieldCondition(key="paper_id", match=MatchValue(value=paper_id))]
    )
    try:
        scroll_result = qdrant.scroll(
            collection_name="gcn_collection",
            scroll_filter=source_filter,
            limit=1,
            with_payload=True,
            with_vectors=True,
        )
        source_points = scroll_result[0] if isinstance(scroll_result, tuple) else scroll_result
        if not source_points or not source_points[0].vector:
            return []

        if hasattr(qdrant, "query_points"):
            hits = qdrant.query_points(
                collection_name="gcn_collection",
                query=source_points[0].vector,
                limit=top_k + 5,
                with_payload=True,
            ).points
        else:
            hits = qdrant.search(
                collection_name="gcn_collection",
                query_vector=source_points[0].vector,
                limit=top_k + 5,
                with_payload=True,
            )
    except Exception as exc:
        logger.exception("Unable to recommend papers for %s: %s", paper_id, exc)
        return []

    recommendations, seen_ids = [], {paper_id}
    for hit in hits:
        payload = hit.payload or {}
        hit_paper_id = str(payload.get("paper_id", ""))
        if not hit_paper_id or hit_paper_id in seen_ids:
            continue
        seen_ids.add(hit_paper_id)
        recommendations.append(
            {
                "paper_id": hit_paper_id,
                "title": payload.get("title", "Khong co tieu de"),
                "year": payload.get("year", "N/A"),
                "gcn_score": float(hit.score),
            }
        )
        if len(recommendations) >= top_k:
            break
    return recommendations
