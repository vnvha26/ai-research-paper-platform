import os
import json
import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.nn import GCNConv
from sentence_transformers import SentenceTransformer
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE      = BASE_DIR / "data/02_processed/papers_100k_complete.jsonl"
EMBEDDING_CACHE = BASE_DIR / "models/node_embeddings.pt"
MODEL_SAVE_PATH = BASE_DIR / "models/gcn_model.pth"
FINAL_VECTORS   = BASE_DIR / "models/gcn_final_embeddings.pt"
METADATA_JSON   = BASE_DIR / "data/s2_processed/papers_metadata.json"
PLOT_IMAGE      = BASE_DIR / "models/gcn_training_metrics.png"


EMBEDDING_CACHE.parent.mkdir(parents=True, exist_ok=True)
MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
METADATA_JSON.parent.mkdir(parents=True, exist_ok=True)

class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index)
        return x

class EdgeDecoder(torch.nn.Module):
    def __init__(self):
        super(EdgeDecoder, self).__init__()

    def forward(self, z, edge_label_index):
        src_node_vector = z[edge_label_index[0]]
        tgt_node_vector = z[edge_label_index[1]]
        return (src_node_vector * tgt_node_vector).sum(dim=-1)

def load_and_build_graph():
    print("\n[1/4] Đang đọc file dữ liệu")
    papers = []
    hash_to_idx = {}
    
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Không tìm thấy file: {INPUT_FILE}")

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            try:
                data = json.loads(line)
                p_id = data.get("paperId") 
                if not p_id: continue
                
                p_id = str(p_id)
                if p_id not in hash_to_idx:
                    idx = len(papers)
                    hash_to_idx[p_id] = idx
                    
                    title = data.get("title", "")
                    abstract = data.get("abstract", "")
                    sec = data.get("sections", {})
                    
                    full_text = ". ".join(
                        text.strip()
                        for text in (
                            title,
                            abstract,
                            sec.get("intro", ""),
                            sec.get("method", ""),
                            sec.get("conclusion", ""),
                        )
                        if text and text.strip()
                    )
                    papers.append({
                        "idx": idx,
                        "id": p_id,
                        "title": title,
                        "text": full_text[:3000], 
                        "citations": data.get("outCitations", [])
                    })
            except Exception as e:
                continue

    num_papers = len(papers)
    if num_papers == 0:
        raise ValueError("Không có bài báo hợp lệ để huấn luyện GCN.")
    print(f"  -> Đã nạp {num_papers} bài báo hợp lệ.")

    print("\n[2/4] Đang chuyển đổi Văn bản thành Vector (Node Features)...")
    need_recompute = True
    if os.path.exists(EMBEDDING_CACHE):
        x = torch.load(EMBEDDING_CACHE, weights_only=True)
        if x.shape[0] == num_papers:
            print("  -> Đã tìm thấy Cache Vector khớp với dữ liệu hiện tại. Đang load...")
            need_recompute = False
        else:
            print(f"  -> Dữ liệu đã thay đổi (từ {x.shape[0]} lên {num_papers} bài). Đang tính lại Vector...")

    if need_recompute:
        print("  -> Đang khởi chạy Mô hình ngôn ngữ (SentenceTransformer)...")
        encoder = SentenceTransformer('all-MiniLM-L6-v2') 
        texts = [p["text"] for p in papers]
        
        embeddings = encoder.encode(texts, show_progress_bar=True, batch_size=128)
        x = torch.tensor(embeddings, dtype=torch.float)
        torch.save(x, EMBEDDING_CACHE)
        print(f"  -> Đã cập nhật và lưu Cache tại: {EMBEDDING_CACHE}")

    print("\n[3/4] Đang xây dựng Ma trận Liên kết (Edges)...")
    sources, targets = [], []
    for p in papers:
        src = p["idx"]
        for c_id in p["citations"]:
            c_id = str(c_id)
            if c_id in hash_to_idx: 
                tgt = hash_to_idx[c_id]
                sources.append(src)
                targets.append(tgt)

    edge_index = torch.tensor([sources, targets], dtype=torch.long)
    
    if edge_index.size(1) < 20:
        print(f"Dữ liệu hiện tại chỉ có {edge_index.size(1)} liên kết trích dẫn chéo thực tế.")
        print(f"Hệ thống tự động tạo giả 200 liên kết để Test Training.")
        
        dummy_sources = torch.randint(0, num_papers, (200,))
        dummy_targets = torch.randint(0, num_papers, (200,))
        edge_index = torch.cat([edge_index, torch.stack([dummy_sources, dummy_targets], dim=0)], dim=1)
    else:
        print(f"  -> Tổng số kết nối trích dẫn thực tế tìm được: {edge_index.size(1)}")

    graph_data = Data(x=x, edge_index=edge_index)
    return graph_data, papers

def train():
    graph_data, papers_info = load_and_build_graph()

    print("\n[4/4] Đang chia tập dữ liệu (Train/Validation/Test)...")
    transform = RandomLinkSplit(is_undirected=False, 
                                num_val=0.1, 
                                num_test=0.1,
                                add_negative_train_samples=True)
    
    train_data, val_data, test_data = transform(graph_data)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  -> Hệ thống đang sử dụng: {device}")

    in_channels = graph_data.num_features 
    hidden_channels = 128
    out_channels = 64 

    encoder = GCNEncoder(in_channels, hidden_channels, out_channels).to(device)
    decoder = EdgeDecoder().to(device)

    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.01)
    criterion = torch.nn.BCEWithLogitsLoss() 

    def train_epoch(data):
        encoder.train()
        optimizer.zero_grad()
        z = encoder(data.x.to(device), data.edge_index.to(device))
        out = decoder(z, data.edge_label_index.to(device))
        loss = criterion(out, data.edge_label.to(device).float())
        loss.backward()
        optimizer.step()
        return loss.item()

    @torch.no_grad()
    def test_epoch(data):
        encoder.eval()
        z = encoder(data.x.to(device), data.edge_index.to(device))
        out = decoder(z, data.edge_label_index.to(device))
        preds = torch.sigmoid(out).cpu().numpy()
        labels = data.edge_label.cpu().numpy()
        
        if len(np.unique(labels)) == 1: return 0.5 
        return roc_auc_score(labels, preds)

    print("\n" + "="*50)
    print("Bắt đầu")
    
    epochs = 1000
    history_loss = []
    history_auc = []

    for epoch in range(1, epochs + 1):
        loss = train_epoch(train_data)
        val_auc = test_epoch(val_data)
        
        history_loss.append(loss)
        history_auc.append(val_auc)
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch: {epoch:03d} | Loss: {loss:.4f} | Val AUC: {val_auc:.4f}")

    test_auc = test_epoch(test_data)
    print(f"\nHOÀN THÀNH HUẤN LUYỆN! ĐIỂM TEST AUC: {test_auc:.4f}")

    torch.save({'encoder': encoder.state_dict(), 'decoder': decoder.state_dict()}, MODEL_SAVE_PATH)
    
    encoder.eval()
    with torch.no_grad():
        final_embeddings = encoder(graph_data.x.to(device), graph_data.edge_index.to(device)).cpu()
        torch.save(final_embeddings, FINAL_VECTORS)
        print(f"Đã lưu Vector GCN tại: {FINAL_VECTORS}")
        
    with open(METADATA_JSON, "w", encoding="utf-8") as f:
        json.dump(papers_info, f, ensure_ascii=False)
        print(f"Đã lưu Metadata tại: {METADATA_JSON}")

    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history_loss, label='Training Loss', color='red', linewidth=2)
    plt.title('GCN Training Loss over Epochs', fontsize=14)
    plt.xlabel('Epochs')
    plt.ylabel('Binary Cross-Entropy Loss')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history_auc, label='Validation AUC', color='blue', linewidth=2)
    plt.axhline(y=test_auc, color='green', linestyle='--', label=f'Final Test AUC ({test_auc:.2f})')
    plt.title('GCN Validation AUC over Epochs', fontsize=14)
    plt.xlabel('Epochs')
    plt.ylabel('ROC-AUC Score')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    plt.tight_layout()
    plt.savefig(PLOT_IMAGE, dpi=300) 
    print(f"Đã xuất Biểu đồ Huấn luyện HD tại: {PLOT_IMAGE}")

if __name__ == "__main__":
    train()
