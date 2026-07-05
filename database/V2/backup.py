import json
import torch
import uuid
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parents[2]
INPUT_FILE = BASE_DIR / "data/02_processed/papers_100k_complete.jsonl"
GCN_VECTORS = BASE_DIR / "models/gcn_final_embeddings.pt"
TEXT_VECTORS = BASE_DIR / "models/node_embeddings.pt"
BATCH_SIZE = 1000

def init_qdrant(client):
    """Tạo bảng Vector an toàn (Fixed Warning)"""
    print("  [+] Đang thiết lập Qdrant Collections...")
    collections = {
        "gcn_collection": 64,
        "dense_collection": 384,
        "rag_chunk_collection": 384
    }
    for name, size in collections.items():
        if not client.collection_exists(name):
            client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=size, distance=Distance.COSINE)
            )

def main():
    print("🚀 BẮT ĐẦU MIGRATION VÀO QDRANT...")
    
    try:
        qdrant = QdrantClient(url="http://localhost:6333")
        init_qdrant(qdrant)
    except Exception as e:
        print(f"❌ Lỗi kết nối Qdrant: {e}")
        return

    print("\n[*] Đang nạp Model AI và Cache Vectors...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    nlp_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    
    gcn_emb = torch.load(GCN_VECTORS, map_location='cpu', weights_only=True).numpy()
    txt_emb = torch.load(TEXT_VECTORS, map_location='cpu', weights_only=True).numpy()

    qdrant_gcn_batch, qdrant_dense_batch = [], []
    rag_ids, rag_texts, rag_payloads = [], [], []

    print("[*] Đang đếm số lượng bài báo...")
    total_lines = sum(1 for _ in open(INPUT_FILE, 'r', encoding='utf-8'))
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(tqdm(f, total=total_lines, desc="Đang nhúng Vector vào Qdrant")):
            if not line.strip(): continue
            p = json.loads(line)
            
            pid = str(p.get("corpusId") or p.get("corpusid") or p.get("paperId"))
            title = p.get("title", "")
            year = p.get("year")
            sections = p.get("sections", {})

            payload = {"paper_id": pid, "title": title, "year": year}
            
            qdrant_gcn_batch.append(PointStruct(id=idx, vector=gcn_emb[idx].tolist(), payload=payload))
            qdrant_dense_batch.append(PointStruct(id=idx, vector=txt_emb[idx].tolist(), payload=payload))

            for sec_name in ["intro", "method", "conclusion"]:
                sec_text = sections.get(sec_name, "").strip()
                if sec_text:
                    chunk_uuid = str(uuid.uuid5(uuid.NAMESPACE_OID, f"{pid}_{sec_name}"))
                    rag_ids.append(chunk_uuid)
                    rag_texts.append(sec_text)
                    rag_payloads.append({"paper_id": pid, "section": sec_name, "title": title})

            if len(qdrant_gcn_batch) >= BATCH_SIZE:
                qdrant.upsert(collection_name="gcn_collection", points=qdrant_gcn_batch)
                qdrant.upsert(collection_name="dense_collection", points=qdrant_dense_batch)
                
                if rag_texts:
                    rag_vectors = nlp_model.encode(rag_texts, batch_size=256, show_progress_bar=False)
                    rag_points = [PointStruct(id=r_id, vector=v.tolist(), payload=pLoad) for r_id, v, pLoad in zip(rag_ids, rag_vectors, rag_payloads)]
                    qdrant.upsert(collection_name="rag_chunk_collection", points=rag_points)

                qdrant_gcn_batch.clear(); qdrant_dense_batch.clear()
                rag_ids.clear(); rag_texts.clear(); rag_payloads.clear()

    if qdrant_gcn_batch:
        qdrant.upsert(collection_name="gcn_collection", points=qdrant_gcn_batch)
        qdrant.upsert(collection_name="dense_collection", points=qdrant_dense_batch)
        if rag_texts:
            rag_vectors = nlp_model.encode(rag_texts, batch_size=256, show_progress_bar=False)
            rag_points = [PointStruct(id=r_id, vector=v.tolist(), payload=pLoad) for r_id, v, pLoad in zip(rag_ids, rag_vectors, rag_payloads)]
            qdrant.upsert(collection_name="rag_chunk_collection", points=rag_points)

    print("\n🎉 XONG! QDRANT ĐÃ NHẬN TOÀN BỘ VECTORS.")

if __name__ == "__main__":
    main()
