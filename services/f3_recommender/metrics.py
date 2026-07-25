import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from services.f3_recommender.config import (
    MAX_RANKING_EVALUATION_EDGES,
    RANKING_NEGATIVES_PER_POSITIVE,
)
from services.f3_recommender.sampling import sample_negative_targets, sample_positive_edges


@torch.no_grad()
def binary_metrics(encoder, decoder, message_passing_graph, positive_edges, metadata):
    normalized_features, years, field_ids, sorted_hashes = metadata
    encoder.eval()
    embeddings = encoder(message_passing_graph.x, message_passing_graph.edge_index)
    sources, targets = positive_edges
    negative_targets = sample_negative_targets(
        sources, targets, normalized_features, years, field_ids, sorted_hashes, hard=False
    ).squeeze(1)
    negative_edges = torch.stack((sources, negative_targets))
    logits = torch.cat((decoder(embeddings, positive_edges), decoder(embeddings, negative_edges)))
    labels = torch.cat((torch.ones(sources.numel(), device=logits.device), torch.zeros(sources.numel(), device=logits.device)))
    probabilities = torch.sigmoid(logits).cpu().numpy()
    labels_np = labels.cpu().numpy()
    return {
        "auc": float(roc_auc_score(labels_np, probabilities)),
        "average_precision": float(average_precision_score(labels_np, probabilities)),
    }


@torch.no_grad()
def ranking_metrics(encoder, decoder, message_passing_graph, positive_edges, metadata):
    """Compare F3 and MiniLM on one held-out citation plus 99 hard negatives."""
    normalized_features, years, field_ids, sorted_hashes = metadata
    encoder.eval()
    embeddings = encoder(message_passing_graph.x, message_passing_graph.edge_index)
    evaluation_edges = sample_positive_edges(positive_edges, MAX_RANKING_EVALUATION_EDGES)
    ranks, baseline_ranks = [], []

    for start in range(0, evaluation_edges.size(1), 128):
        edges = evaluation_edges[:, start:start + 128]
        sources, targets = edges
        negative_targets = sample_negative_targets(
            sources,
            targets,
            normalized_features,
            years,
            field_ids,
            sorted_hashes,
            negatives_per_positive=RANKING_NEGATIVES_PER_POSITIVE,
            candidate_multiplier=3,
            hard=True,
        )
        negative_edges = torch.stack(
            (
                sources.unsqueeze(1).expand_as(negative_targets).reshape(-1),
                negative_targets.reshape(-1),
            )
        )
        positive_scores = decoder(embeddings, edges)
        negative_scores = decoder(embeddings, negative_edges).reshape_as(negative_targets)
        ranks.append(1 + (negative_scores >= positive_scores.unsqueeze(1)).sum(dim=1))

        baseline_positive = (normalized_features[sources] * normalized_features[targets]).sum(dim=1)
        baseline_negative = (
            normalized_features[sources].unsqueeze(1) * normalized_features[negative_targets]
        ).sum(dim=-1)
        baseline_ranks.append(1 + (baseline_negative >= baseline_positive.unsqueeze(1)).sum(dim=1))

    return {
        "evaluated_positive_edges": int(sum(rank.numel() for rank in ranks)),
        "gcn": _summarise_ranks(torch.cat(ranks).float()),
        "minilm_baseline": _summarise_ranks(torch.cat(baseline_ranks).float()),
    }


def _summarise_ranks(ranks):
    hits_at_10 = float((ranks <= 10).float().mean().item())
    return {
        "hits_at_10": hits_at_10,
        "recall_at_10": hits_at_10,
        "mrr": float((1.0 / ranks).mean().item()),
    }
