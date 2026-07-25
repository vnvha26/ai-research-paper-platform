import torch

from services.f3_recommender.config import (
    NEGATIVE_SAMPLING_BATCH_SIZE,
    NEGATIVE_SAMPLING_MAX_ATTEMPTS,
)


def edge_hashes(edge_index, num_nodes):
    return torch.unique(edge_index[0].long() * num_nodes + edge_index[1].long(), sorted=True)


def existing_edge_mask(sources, targets, sorted_edge_hashes, num_nodes):
    hashes = sources.long() * num_nodes + targets.long()
    positions = torch.searchsorted(sorted_edge_hashes, hashes)
    in_range = positions < sorted_edge_hashes.numel()
    exists = torch.zeros_like(in_range, dtype=torch.bool)
    exists[in_range] = sorted_edge_hashes[positions[in_range]] == hashes[in_range]
    return exists


def sample_positive_edges(edge_index, max_edges):
    if edge_index.size(1) <= max_edges:
        return edge_index
    chosen = torch.randperm(edge_index.size(1), device=edge_index.device)[:max_edges]
    return edge_index[:, chosen]


def sample_negative_targets(
    sources,
    positive_targets,
    normalized_features,
    years,
    field_ids,
    sorted_edge_hashes,
    negatives_per_positive=1,
    candidate_multiplier=1,
    hard=False,
):
    """Sample valid non-edges, using semantic and metadata similarity when hard."""
    batches = []
    for start in range(0, sources.numel(), NEGATIVE_SAMPLING_BATCH_SIZE):
        batches.append(
            _sample_negative_targets_batch(
                sources[start:start + NEGATIVE_SAMPLING_BATCH_SIZE],
                positive_targets[start:start + NEGATIVE_SAMPLING_BATCH_SIZE],
                normalized_features,
                years,
                field_ids,
                sorted_edge_hashes,
                negatives_per_positive,
                candidate_multiplier,
                hard,
            )
        )
    return torch.cat(batches, dim=0)


def _sample_negative_targets_batch(
    sources,
    positive_targets,
    normalized_features,
    years,
    field_ids,
    sorted_edge_hashes,
    negatives_per_positive,
    candidate_multiplier,
    hard,
):
    device = sources.device
    num_nodes = normalized_features.size(0)
    selected_targets = torch.empty((sources.numel(), negatives_per_positive), dtype=torch.long, device=device)
    unresolved = torch.arange(sources.numel(), device=device)
    base_candidate_count = max(negatives_per_positive, negatives_per_positive * candidate_multiplier)

    for attempt in range(NEGATIVE_SAMPLING_MAX_ATTEMPTS):
        if unresolved.numel() == 0:
            break
        candidate_count = base_candidate_count * (2 ** attempt)
        current_sources = sources[unresolved]
        current_positive_targets = positive_targets[unresolved]
        candidate_targets = torch.randint(num_nodes, (current_sources.numel(), candidate_count), device=device)
        invalid, source_years, target_years, has_years = _invalid_candidate_mask(
            current_sources,
            current_positive_targets,
            candidate_targets,
            years,
            sorted_edge_hashes,
            num_nodes,
        )
        quality = _candidate_quality(
            current_sources,
            candidate_targets,
            normalized_features,
            source_years,
            target_years,
            has_years,
            field_ids,
            hard,
        ).masked_fill(invalid, float("-inf"))
        best_quality, best_indices = torch.topk(quality, k=negatives_per_positive, dim=1)
        resolved = ~torch.isinf(best_quality).any(dim=1)
        if resolved.any():
            selected_targets[unresolved[resolved]] = candidate_targets[resolved].gather(1, best_indices[resolved])
        unresolved = unresolved[~resolved]

    if unresolved.numel():
        raise RuntimeError(
            f"Unable to sample valid historical negative edges for {unresolved.numel():,} papers after "
            f"{NEGATIVE_SAMPLING_MAX_ATTEMPTS} attempts."
        )
    return selected_targets


def _invalid_candidate_mask(sources, positive_targets, candidate_targets, years, sorted_edge_hashes, num_nodes):
    candidate_sources = sources.unsqueeze(1).expand_as(candidate_targets)
    invalid = candidate_sources == candidate_targets
    invalid |= candidate_targets == positive_targets.unsqueeze(1)
    invalid |= existing_edge_mask(
        candidate_sources.reshape(-1),
        candidate_targets.reshape(-1),
        sorted_edge_hashes,
        num_nodes,
    ).reshape_as(candidate_targets)
    source_years = years[sources].unsqueeze(1)
    target_years = years[candidate_targets]
    has_years = (source_years > 0) & (target_years > 0)
    invalid |= has_years & (target_years > source_years)
    return invalid, source_years, target_years, has_years


def _candidate_quality(
    sources,
    candidate_targets,
    normalized_features,
    source_years,
    target_years,
    has_years,
    field_ids,
    hard,
):
    if not hard:
        return torch.rand(candidate_targets.shape, device=sources.device)
    quality = (
        normalized_features[sources].unsqueeze(1) * normalized_features[candidate_targets]
    ).sum(dim=-1)
    same_field = (
        (field_ids[sources].unsqueeze(1) > 0)
        & (field_ids[sources].unsqueeze(1) == field_ids[candidate_targets])
    )
    close_year = has_years & ((source_years - target_years).abs() <= 5)
    return quality + same_field.float() * 0.10 + close_year.float() * 0.05
