import json

import torch
from sentence_transformers import SentenceTransformer
from torch_geometric.data import Data

from services.f3_recommender.config import (
    EMBEDDING_CACHE,
    INPUT_FILE,
    TRAIN_END_YEAR,
    VALIDATION_END_YEAR,
)


def _as_year(value):
    try:
        year = int(value)
        return year if 1900 <= year <= 2100 else 0
    except (TypeError, ValueError):
        return 0


def _extract_fields(paper):
    raw_fields = (
        paper.get("s2FieldsOfStudy")
        or paper.get("fieldsOfStudy")
        or paper.get("fields")
        or []
    )
    if isinstance(raw_fields, str):
        return [raw_fields.strip()] if raw_fields.strip() else []
    if not isinstance(raw_fields, list):
        return []

    names = []
    for item in raw_fields:
        if isinstance(item, str):
            name = item
        elif isinstance(item, dict):
            name = item.get("category") or item.get("name") or item.get("field")
        else:
            name = None
        if name and str(name).strip():
            names.append(str(name).strip())
    return sorted(set(names))


def load_and_build_graph():
    """Load papers, reuse/create MiniLM features, and retain only true internal citations."""
    print("\n[1/4] Loading papers and verified citation edges...")
    papers = []
    paper_id_to_index = {}
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    with INPUT_FILE.open("r", encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            paper_id = record.get("paperId")
            if not paper_id:
                continue
            paper_id = str(paper_id)
            if paper_id in paper_id_to_index:
                continue

            index = len(papers)
            paper_id_to_index[paper_id] = index
            sections = record.get("sections") or {}
            full_text = ". ".join(
                text.strip()
                for text in (
                    record.get("title", ""),
                    record.get("abstract", ""),
                    sections.get("intro", ""),
                    sections.get("method", ""),
                    sections.get("conclusion", ""),
                )
                if isinstance(text, str) and text.strip()
            )
            papers.append(
                {
                    "idx": index,
                    "id": paper_id,
                    "title": record.get("title", ""),
                    "text": full_text[:3000],
                    "citations": record.get("outCitations", []) or [],
                    "year": _as_year(record.get("year")),
                    "fields": _extract_fields(record),
                }
            )

    if not papers:
        raise ValueError("No valid papers were found for GCN training.")

    print(f"  -> Loaded {len(papers):,} papers.")
    features = _load_or_create_embeddings(papers)
    edge_index = _build_internal_citation_edges(papers, paper_id_to_index)
    return Data(x=features, edge_index=edge_index), papers


def _load_or_create_embeddings(papers):
    print("\n[2/4] Loading or creating text embeddings...")
    features = None
    if EMBEDDING_CACHE.exists():
        features = torch.load(EMBEDDING_CACHE, weights_only=True)
        if features.shape[0] != len(papers):
            print("  -> Cached embeddings do not match the input; recomputing.")
            features = None
        else:
            print("  -> Reusing cached text embeddings.")

    if features is None:
        encoder = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = encoder.encode(
            [paper["text"] for paper in papers],
            show_progress_bar=True,
            batch_size=128,
        )
        features = torch.tensor(embeddings, dtype=torch.float)
        EMBEDDING_CACHE.parent.mkdir(parents=True, exist_ok=True)
        torch.save(features, EMBEDDING_CACHE)
    return features


def _build_internal_citation_edges(papers, paper_id_to_index):
    print("\n[3/4] Building the graph from real, internal citations only...")
    sources, targets = [], []
    for paper in papers:
        source_index = paper["idx"]
        for cited_id in paper["citations"]:
            target_index = paper_id_to_index.get(str(cited_id))
            if target_index is not None and target_index != source_index:
                sources.append(source_index)
                targets.append(target_index)

    if not sources:
        raise ValueError("No internal citation edges were found; fake edges are not created.")
    raw_edges = torch.tensor([sources, targets], dtype=torch.long)
    edge_index = torch.unique(raw_edges, dim=1)
    print(
        f"  -> {edge_index.size(1):,} unique real citation edges "
        f"({raw_edges.size(1) - edge_index.size(1):,} duplicate edges removed)."
    )
    return edge_index


def build_node_metadata(papers):
    """Encode publication year and primary field for negative sampling."""
    field_to_id = {}
    field_ids = []
    for paper in papers:
        primary_field = paper["fields"][0] if paper["fields"] else ""
        if primary_field and primary_field not in field_to_id:
            field_to_id[primary_field] = len(field_to_id) + 1
        field_ids.append(field_to_id.get(primary_field, 0))
    return (
        torch.tensor([paper["year"] for paper in papers], dtype=torch.long),
        torch.tensor(field_ids, dtype=torch.long),
    )


def split_edges_temporally(edge_index, years):
    """Split citations by the publication year of the citing paper."""
    source_years = years[edge_index[0]]
    target_years = years[edge_index[1]]
    valid_temporal_edge = (source_years > 0) & (target_years > 0) & (target_years <= source_years)
    temporal_edges = edge_index[:, valid_temporal_edge]
    temporal_years = source_years[valid_temporal_edge]
    dropped = edge_index.size(1) - temporal_edges.size(1)

    train_mask = temporal_years <= TRAIN_END_YEAR
    validation_mask = (temporal_years > TRAIN_END_YEAR) & (temporal_years <= VALIDATION_END_YEAR)
    test_mask = temporal_years > VALIDATION_END_YEAR

    minimum_split_size = min(100, max(10, temporal_edges.size(1) // 50))
    if validation_mask.sum() < minimum_split_size or test_mask.sum() < minimum_split_size:
        train_mask, validation_mask, test_mask = _fallback_temporal_masks(temporal_years)

    train_edges = temporal_edges[:, train_mask]
    validation_edges = temporal_edges[:, validation_mask]
    test_edges = temporal_edges[:, test_mask]
    if min(train_edges.size(1), validation_edges.size(1), test_edges.size(1)) == 0:
        raise ValueError("Temporal split produced an empty train, validation, or test set.")

    print("\n[4/4] Temporal split (citation year = citing paper year):")
    print(f"  -> Train edges:      {train_edges.size(1):,}")
    print(f"  -> Validation edges: {validation_edges.size(1):,}")
    print(f"  -> Test edges:       {test_edges.size(1):,}")
    print(f"  -> Dropped invalid/future citation edges: {dropped:,}")
    return train_edges, validation_edges, test_edges


def _fallback_temporal_masks(temporal_years):
    available_years = sorted(set(temporal_years.tolist()))
    if len(available_years) < 3:
        raise ValueError("At least three publication years are required for temporal evaluation.")
    train_boundary = available_years[max(0, int(len(available_years) * 0.7) - 1)]
    validation_boundary = available_years[
        min(len(available_years) - 2, max(1, int(len(available_years) * 0.85) - 1))
    ]
    print(
        "  -> Default temporal years were sparse; using ordered year boundaries "
        f"{train_boundary} / {validation_boundary}."
    )
    return (
        temporal_years <= train_boundary,
        (temporal_years > train_boundary) & (temporal_years <= validation_boundary),
        temporal_years > validation_boundary,
    )
