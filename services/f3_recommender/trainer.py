import json

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from services.f3_recommender.config import (
    EARLY_STOPPING_PATIENCE,
    FINAL_VECTORS,
    HARD_NEGATIVE_CANDIDATES,
    HIDDEN_CHANNELS,
    LEARNING_RATE,
    MAX_EPOCHS,
    MAX_TRAIN_EDGES_PER_EPOCH,
    METADATA_JSON,
    MIN_DELTA,
    MODEL_SAVE_PATH,
    OUTPUT_CHANNELS,
    PLOT_IMAGE,
    RANDOM_SEED,
    REPORT_JSON,
    WEIGHT_DECAY,
)
from services.f3_recommender.data import (
    build_node_metadata,
    load_and_build_graph,
    split_edges_temporally,
)
from services.f3_recommender.metrics import binary_metrics, ranking_metrics
from services.f3_recommender.model import EdgeDecoder, GCNEncoder
from services.f3_recommender.sampling import edge_hashes, sample_negative_targets, sample_positive_edges


def train():
    """Train F3, select its best checkpoint, evaluate it, and save artifacts."""
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    _ensure_output_directories()

    graph_data, papers = load_and_build_graph()
    years, field_ids = build_node_metadata(papers)
    train_edges, validation_edges, test_edges = split_edges_temporally(graph_data.edge_index, years)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nTraining on: {device}")
    graph_data = graph_data.to(device)
    train_edges, validation_edges, test_edges = (
        train_edges.to(device),
        validation_edges.to(device),
        test_edges.to(device),
    )
    training_graph = Data(x=graph_data.x, edge_index=train_edges)
    metadata = (
        F.normalize(graph_data.x, p=2, dim=1),
        years.to(device),
        field_ids.to(device),
        edge_hashes(graph_data.edge_index, graph_data.num_nodes).to(device),
    )

    encoder = GCNEncoder(graph_data.num_features, HIDDEN_CHANNELS, OUTPUT_CHANNELS).to(device)
    decoder = EdgeDecoder().to(device)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    criterion = torch.nn.BCEWithLogitsLoss()
    best_epoch, best_auc, history = _fit(
        encoder,
        decoder,
        optimizer,
        criterion,
        training_graph,
        train_edges,
        validation_edges,
        metadata,
        device,
    )

    checkpoint = torch.load(MODEL_SAVE_PATH, map_location=device, weights_only=True)
    encoder.load_state_dict(checkpoint["encoder"])
    decoder.load_state_dict(checkpoint["decoder"])
    test_binary = binary_metrics(encoder, decoder, training_graph, test_edges, metadata)
    test_ranking = ranking_metrics(encoder, decoder, training_graph, test_edges, metadata)
    _print_test_metrics(test_binary, test_ranking)
    _save_artifacts(
        encoder,
        graph_data,
        papers,
        best_epoch,
        best_auc,
        test_binary,
        test_ranking,
        train_edges,
        validation_edges,
        test_edges,
        history,
    )


def _fit(encoder, decoder, optimizer, criterion, training_graph, train_edges, validation_edges, metadata, device):
    best_auc, best_epoch, no_improvement = float("-inf"), 0, 0
    history = {"loss": [], "validation_auc": []}
    print("\nStarting temporal link-prediction training...")

    for epoch in range(1, MAX_EPOCHS + 1):
        loss, sampled_edges = _train_epoch(
            encoder, decoder, optimizer, criterion, training_graph, train_edges, metadata, device
        )
        validation = binary_metrics(encoder, decoder, training_graph, validation_edges, metadata)
        history["loss"].append(loss)
        history["validation_auc"].append(validation["auc"])

        if validation["auc"] > best_auc + MIN_DELTA:
            best_auc, best_epoch, no_improvement = validation["auc"], epoch, 0
            _save_checkpoint(encoder, decoder, epoch, validation, training_graph.num_features)
        else:
            no_improvement += 1

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Epoch {epoch:03d} | loss {loss:.4f} | sampled train edges {sampled_edges:,} | "
                f"val AUC {validation['auc']:.4f} | val AP {validation['average_precision']:.4f}"
            )
        if no_improvement >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping at epoch {epoch}; best validation AUC was at epoch {best_epoch}.")
            break
    return best_epoch, best_auc, history


def _train_epoch(encoder, decoder, optimizer, criterion, training_graph, train_edges, metadata, device):
    encoder.train()
    optimizer.zero_grad()
    positive_edges = sample_positive_edges(train_edges, MAX_TRAIN_EDGES_PER_EPOCH)
    negative_targets = sample_negative_targets(
        positive_edges[0],
        positive_edges[1],
        *metadata,
        candidate_multiplier=HARD_NEGATIVE_CANDIDATES,
        hard=True,
    ).squeeze(1)
    negative_edges = torch.stack((positive_edges[0], negative_targets))
    embeddings = encoder(training_graph.x, training_graph.edge_index)
    logits = torch.cat((decoder(embeddings, positive_edges), decoder(embeddings, negative_edges)))
    labels = torch.cat(
        (torch.ones(positive_edges.size(1), device=device), torch.zeros(positive_edges.size(1), device=device))
    )
    loss = criterion(logits, labels)
    loss.backward()
    optimizer.step()
    return float(loss.item()), int(positive_edges.size(1))


def _save_checkpoint(encoder, decoder, epoch, validation, input_size):
    torch.save(
        {
            "encoder": encoder.state_dict(),
            "decoder": decoder.state_dict(),
            "epoch": epoch,
            "validation": validation,
            "model_dimensions": {
                "input": input_size,
                "hidden": HIDDEN_CHANNELS,
                "output": OUTPUT_CHANNELS,
            },
        },
        MODEL_SAVE_PATH,
    )


def _print_test_metrics(test_binary, test_ranking):
    print("\nTest metrics:")
    print(f"  -> ROC-AUC: {test_binary['auc']:.4f} | Average Precision: {test_binary['average_precision']:.4f}")
    print(
        "  -> GCN sampled ranking: "
        f"Hits@10 {test_ranking['gcn']['hits_at_10']:.4f} | MRR {test_ranking['gcn']['mrr']:.4f}"
    )
    print(
        "  -> MiniLM baseline: "
        f"Hits@10 {test_ranking['minilm_baseline']['hits_at_10']:.4f} | "
        f"MRR {test_ranking['minilm_baseline']['mrr']:.4f}"
    )


def _save_artifacts(encoder, graph_data, papers, best_epoch, best_auc, test_binary, test_ranking, train_edges, validation_edges, test_edges, history):
    encoder.eval()
    with torch.no_grad():
        torch.save(encoder(graph_data.x, graph_data.edge_index).cpu(), FINAL_VECTORS)

    serialisable_papers = [{key: value for key, value in paper.items() if key != "text"} for paper in papers]
    with METADATA_JSON.open("w", encoding="utf-8") as metadata_file:
        json.dump(serialisable_papers, metadata_file, ensure_ascii=False)

    report = {
        "best_epoch": best_epoch,
        "best_validation_auc": best_auc,
        "test_binary": test_binary,
        "test_ranking": test_ranking,
        "ranking_note": "Hits@10, Recall@10 and MRR use one held-out citation against 99 sampled hard non-citations.",
        "temporal_split": {
            "train_edges": int(train_edges.size(1)),
            "validation_edges": int(validation_edges.size(1)),
            "test_edges": int(test_edges.size(1)),
        },
    }
    with REPORT_JSON.open("w", encoding="utf-8") as report_file:
        json.dump(report, report_file, ensure_ascii=False, indent=2)
    _save_training_plot(history, test_binary["auc"])
    print(f"\nSaved best model: {MODEL_SAVE_PATH}")
    print(f"Saved deployment embeddings: {FINAL_VECTORS}")
    print(f"Saved training report: {REPORT_JSON}")


def _save_training_plot(history, test_auc):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history["loss"], label="Training loss", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Binary cross-entropy")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.subplot(1, 2, 2)
    plt.plot(history["validation_auc"], label="Validation ROC-AUC", color="blue")
    plt.axhline(y=test_auc, color="green", linestyle="--", label="Test ROC-AUC")
    plt.xlabel("Epoch")
    plt.ylabel("ROC-AUC")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(PLOT_IMAGE, dpi=300)


def _ensure_output_directories():
    for output_dir in (MODEL_SAVE_PATH.parent, METADATA_JSON.parent):
        output_dir.mkdir(parents=True, exist_ok=True)
