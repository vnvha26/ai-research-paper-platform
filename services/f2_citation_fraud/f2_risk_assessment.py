from services.f2_citation_fraud.f2_fraud_config import (
    COMMON_NEIGHBOR_THRESHOLD,
    LOCAL_DENSITY_MIN_JACCARD,
    LOCAL_DENSITY_THRESHOLD,
    NEIGHBOR_JACCARD_THRESHOLD,
)


def is_abnormal_neighborhood(neighbor_jaccard, common_neighbor_count, local_density):
    dense_overlap = (
        neighbor_jaccard >= NEIGHBOR_JACCARD_THRESHOLD
        and common_neighbor_count >= COMMON_NEIGHBOR_THRESHOLD
    )
    dense_local_cluster = (
        local_density >= LOCAL_DENSITY_THRESHOLD
        and common_neighbor_count >= COMMON_NEIGHBOR_THRESHOLD
        and neighbor_jaccard >= LOCAL_DENSITY_MIN_JACCARD
    )
    return dense_overlap or dense_local_cluster


def status_from_signals(low_relevance, is_mutual, has_shared_author, graph_abnormal):
    if low_relevance and is_mutual and has_shared_author and graph_abnormal:
        return 2, "Cờ đỏ: cite chéo + chung tác giả + cụm citation dày + nội dung ít liên quan"
    if low_relevance and graph_abnormal and has_shared_author:
        return 2, "Cờ đỏ: chung tác giả + cụm citation dày nhưng nội dung ít liên quan"
    if low_relevance and graph_abnormal:
        return 2, "Cờ đỏ: cụm citation dày nhưng nội dung ít liên quan"
    if low_relevance and is_mutual and has_shared_author:
        return 2, "Cờ đỏ: cite chéo + chung tác giả + nội dung ít liên quan"
    if low_relevance and is_mutual:
        return 2, "Cờ đỏ: cite chéo nhưng nội dung ít liên quan"
    if low_relevance and has_shared_author:
        return 1, "Cờ vàng: tự/nhóm trích dẫn nhưng nội dung ít liên quan"
    if low_relevance:
        return 1, "Cờ vàng: trích dẫn nội dung ít liên quan"
    if graph_abnormal and has_shared_author:
        return 1, "Cờ vàng: chung tác giả và cụm citation dày"
    if graph_abnormal:
        return 1, "Cờ vàng: cụm citation/hàng xóm chung dày bất thường"
    if is_mutual:
        return 1, "Cờ vàng: trích dẫn chéo"
    return 0, "Hợp lệ"


def graph_metrics(record):
    common_count = int(record.get("common_neighbor_count") or 0)
    union_count = int(record.get("union_neighbor_count") or 0)
    local_node_count = int(record.get("local_node_count") or 0)
    local_edge_count = int(record.get("local_edge_count") or 0)
    neighbor_jaccard = common_count / union_count if union_count else 0.0
    max_local_edges = local_node_count * (local_node_count - 1)
    local_density = local_edge_count / max_local_edges if max_local_edges > 0 else 0.0
    return {
        "common_neighbor_count": common_count,
        "union_neighbor_count": union_count,
        "local_node_count": local_node_count,
        "local_edge_count": local_edge_count,
        "neighbor_jaccard": neighbor_jaccard,
        "local_density": local_density,
        "graph_abnormal": is_abnormal_neighborhood(
            neighbor_jaccard,
            common_count,
            local_density,
        ),
    }
