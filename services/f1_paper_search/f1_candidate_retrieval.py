import math

try:
    from qdrant_client.models import FieldCondition, Filter, MatchAny, QueryRequest
except ModuleNotFoundError:
    FieldCondition = None
    Filter = None
    MatchAny = None
    QueryRequest = None

from services.f1_paper_search.f1_search_config import MULTIVEC_COLLECTION, SECTION_WEIGHTS
from services.f1_paper_search.f1_query_ranking import build_tsquery


def lexical_search(pg_conn, keywords: list[str], top_k: int = 500) -> list[dict]:
    if not keywords:
        return []

    cursor = pg_conn.cursor()
    sql = """
        SELECT paper_id, title, year, abstract
        FROM papers
        WHERE to_tsvector('english', title) @@ to_tsquery('english', %s)
        LIMIT %s;
    """
    rows = []
    try:
        cursor.execute(sql, (build_tsquery(keywords, "AND"), top_k))
        rows = cursor.fetchall()
    except Exception:
        pg_conn.rollback()

    if not rows:
        try:
            cursor.execute(sql, (build_tsquery(keywords, "OR"), top_k))
            rows = cursor.fetchall()
        except Exception as exc:
            pg_conn.rollback()
            raise RuntimeError(f"Lỗi tìm kiếm PostgreSQL: {exc}") from exc
        finally:
            cursor.close()
    else:
        cursor.close()

    return [
        {"paper_id": row[0], "title": row[1], "year": row[2], "abstract": row[3] or ""}
        for row in rows
    ]


def semantic_search(qdrant, query_vector: list, limit: int) -> list:
    if hasattr(qdrant, "query_points"):
        return qdrant.query_points(
            collection_name=MULTIVEC_COLLECTION,
            query=query_vector,
            limit=limit,
        ).points
    return qdrant.search(
        collection_name=MULTIVEC_COLLECTION,
        query_vector=query_vector,
        limit=limit,
    )


def get_weighted_cosine_score(
    qdrant,
    query_vector: list,
    paper_id: str,
    section_weights: dict[str, float] | None = None,
) -> float:
    return get_weighted_cosine_scores(
        qdrant,
        query_vector,
        [paper_id],
        section_weights=section_weights,
    ).get(paper_id, 0.0)


def _weighted_score(section_scores, weights):
    total_weight = sum(weights[name] for name in section_scores if name in weights)
    if total_weight == 0:
        return 0.0
    weighted_sum = sum(
        weights[name] * score
        for name, score in section_scores.items()
        if name in weights
    )
    return weighted_sum / total_weight


def get_section_scores_batch(
    qdrant,
    query_vector: list,
    paper_ids: list[str],
    batch_size: int = 64,
) -> dict[str, dict[str, float]]:
    if not paper_ids:
        return {}
    if Filter is None or FieldCondition is None or MatchAny is None:
        return {paper_id: {} for paper_id in paper_ids}
    if hasattr(qdrant, "scroll"):
        return _get_section_scores_by_scroll(
            qdrant,
            query_vector,
            paper_ids,
            batch_size=batch_size,
        )
    if QueryRequest is None or not hasattr(qdrant, "query_batch_points"):
        return {
            paper_id: _get_single_paper_section_scores(qdrant, query_vector, paper_id)
            for paper_id in paper_ids
        }

    all_scores = {}
    try:
        for start in range(0, len(paper_ids), batch_size):
            batch_ids = paper_ids[start:start + batch_size]
            requests = [
                QueryRequest(
                    query=query_vector,
                    filter=Filter(
                        must=[FieldCondition(key="paper_id", match=MatchAny(any=[paper_id]))]
                    ),
                    limit=4,
                    with_payload=["section"],
                )
                for paper_id in batch_ids
            ]
            responses = qdrant.query_batch_points(
                collection_name=MULTIVEC_COLLECTION,
                requests=requests,
            )
            if len(responses) != len(batch_ids):
                raise RuntimeError("Qdrant trả về sai số lượng kết quả batch.")
            for paper_id, response in zip(batch_ids, responses):
                all_scores[paper_id] = {
                    hit.payload.get("section", "core"): hit.score
                    for hit in response.points
                }
    except Exception as exc:
        raise RuntimeError(f"Lỗi batch cosine từ Qdrant: {exc}") from exc
    return all_scores


def _get_section_scores_by_scroll(qdrant, query_vector, paper_ids, batch_size):
    query_norm = math.sqrt(sum(float(value) ** 2 for value in query_vector))
    if query_norm == 0:
        return {paper_id: {} for paper_id in paper_ids}
    all_scores = {paper_id: {} for paper_id in paper_ids}
    try:
        for start in range(0, len(paper_ids), batch_size):
            batch_ids = paper_ids[start:start + batch_size]
            scroll_filter = Filter(
                must=[FieldCondition(key="paper_id", match=MatchAny(any=batch_ids))]
            )
            offset = None
            while True:
                points, next_offset = qdrant.scroll(
                    collection_name=MULTIVEC_COLLECTION,
                    scroll_filter=scroll_filter,
                    limit=max(len(batch_ids) * 4, 1),
                    offset=offset,
                    with_payload=["paper_id", "section"],
                    with_vectors=True,
                )
                for point in points:
                    payload = point.payload or {}
                    paper_id = str(payload.get("paper_id", ""))
                    vector = point.vector
                    if isinstance(vector, dict):
                        vector = next(iter(vector.values()), None)
                    if paper_id not in all_scores or not vector or len(vector) != len(query_vector):
                        continue
                    vector_norm = math.sqrt(sum(float(value) ** 2 for value in vector))
                    if vector_norm == 0:
                        score = 0.0
                    else:
                        dot_product = sum(
                            float(left) * float(right)
                            for left, right in zip(query_vector, vector)
                        )
                        score = dot_product / (query_norm * vector_norm)
                    section = payload.get("section", "core")
                    previous = all_scores[paper_id].get(section)
                    if previous is None or score > previous:
                        all_scores[paper_id][section] = score
                if next_offset is None:
                    break
                offset = next_offset
    except Exception as exc:
        raise RuntimeError(f"Lỗi đọc vector từ Qdrant: {exc}") from exc
    return all_scores


def get_weighted_cosine_scores(
    qdrant,
    query_vector: list,
    paper_ids: list[str],
    section_weights: dict[str, float] | None = None,
) -> dict[str, float]:
    weights = section_weights or SECTION_WEIGHTS
    section_scores = get_section_scores_batch(qdrant, query_vector, paper_ids)
    return {
        paper_id: _weighted_score(scores, weights)
        for paper_id, scores in section_scores.items()
    }


def _get_single_paper_section_scores(qdrant, query_vector, paper_id):
    if Filter is None or FieldCondition is None or MatchAny is None:
        return {}

    search_filter = Filter(
        must=[FieldCondition(key="paper_id", match=MatchAny(any=[paper_id]))]
    )
    try:
        if hasattr(qdrant, "query_points"):
            hits = qdrant.query_points(
                collection_name=MULTIVEC_COLLECTION,
                query=query_vector,
                query_filter=search_filter,
                limit=4,
            ).points
        else:
            hits = qdrant.search(
                collection_name=MULTIVEC_COLLECTION,
                query_vector=query_vector,
                query_filter=search_filter,
                limit=4,
            )
    except Exception as exc:
        raise RuntimeError(f"Lỗi tính cosine từ Qdrant: {exc}") from exc

    return {
        hit.payload.get("section", "core"): hit.score
        for hit in hits
    }


def fetch_sections(pg_conn, paper_ids: list[str]) -> dict[str, dict]:
    if not paper_ids:
        return {}

    cursor = pg_conn.cursor()
    placeholders = ",".join(["%s"] * len(paper_ids))
    try:
        cursor.execute(
            f"SELECT paper_id, abstract, intro_text, method_text, conclusion_text "
            f"FROM papers WHERE paper_id IN ({placeholders})",
            tuple(paper_ids),
        )
        rows = cursor.fetchall()
    except Exception as exc:
        pg_conn.rollback()
        raise RuntimeError(f"Lỗi lấy nội dung PostgreSQL: {exc}") from exc
    finally:
        cursor.close()

    return {
        row[0]: {
            "abstract": row[1] or "",
            "intro": row[2] or "",
            "method": row[3] or "",
            "conclusion": row[4] or "",
        }
        for row in rows
    }
