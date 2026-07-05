import json
import torch
import uuid
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_FILE = BASE_DIR / "data/02_processed/papers_100k_complete.jsonl"
GCN_VECTORS = BASE_DIR / "models/gcn_final_embeddings.pt"
TEXT_VECTORS = BASE_DIR / "models/node_embeddings.pt"

BATCH_SIZE = 1000


def init_qdrant(client):
    """
    Tạo các Qdrant collections cần thiết.
    F1 cũ dùng dense_collection.
    F1_v2 dùng papers_multivec.
    F2/F3 vẫn có thể dùng các collection còn lại.
    """
    print("  [+] Đang thiết lập Qdrant Collections...")

    collections = {
        "gcn_collection": 64,
        "dense_collection": 384,
        "rag_chunk_collection": 384,
        "papers_multivec": 384
    }

    for name, size in collections.items():
        if not client.collection_exists(name):
            client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(
                    size=size,
                    distance=Distance.COSINE
                )
            )
            print(f"      [+] Đã tạo collection: {name}")
        else:
            print(f"      [=] Collection đã tồn tại: {name}")


def make_uuid(text: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_OID, text))


def main():
    print("BẮT ĐẦU MIGRATION VÀO QDRANT...")

    try:
        qdrant = QdrantClient(url="http://localhost:6333")
        init_qdrant(qdrant)
    except Exception as e:
        print(f"❌ Lỗi kết nối Qdrant: {e}")
        return

    print("\n[*] Đang nạp Model AI và Cache Vectors...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Thiết bị đang dùng: {device}")

    nlp_model = SentenceTransformer(
        "all-MiniLM-L6-v2",
        device=device
    )

    try:
        gcn_emb = torch.load(
            GCN_VECTORS,
            map_location="cpu",
            weights_only=True
        ).numpy()

        txt_emb = torch.load(
            TEXT_VECTORS,
            map_location="cpu",
            weights_only=True
        ).numpy()

    except Exception as e:
        print(f"Lỗi load vector .pt: {e}")
        return

    qdrant_gcn_batch = []
    qdrant_dense_batch = []

    rag_ids = []
    rag_texts = []
    rag_payloads = []

    multivec_ids = []
    multivec_texts = []
    multivec_payloads = []

    print("[*] Đang đếm số lượng bài báo...")

    try:
        total_lines = sum(
            1 for _ in open(INPUT_FILE, "r", encoding="utf-8")
        )
    except Exception as e:
        print(f"❌ Không đọc được INPUT_FILE: {e}")
        return

    seen_paper_ids = set()
    vector_index = 0

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for idx, line in enumerate(
            tqdm(f, total=total_lines, desc="Đang nhúng Vector vào Qdrant")
        ):
            if not line.strip():
                continue

            try:
                p = json.loads(line)
            except Exception:
                continue

            raw_pid = p.get("corpusId") or p.get("corpusid") or p.get("paperId")
            if not raw_pid:
                continue
            pid = str(raw_pid)
            source_paper_id = p.get("paperId")
            current_vector_index = None
            if source_paper_id:
                source_paper_id = str(source_paper_id)
                if source_paper_id in seen_paper_ids:
                    continue
                seen_paper_ids.add(source_paper_id)
                current_vector_index = vector_index
                vector_index += 1

            title = p.get("title", "") or ""
            year = p.get("year")
            abstract = p.get("abstract", "") or ""
            sections = p.get("sections", {}) or {}

            payload = {
                "paper_id": pid,
                "title": title,
                "year": year
            }

            if current_vector_index is not None and current_vector_index < len(gcn_emb):
                qdrant_gcn_batch.append(
                    PointStruct(
                        id=current_vector_index,
                        vector=gcn_emb[current_vector_index].tolist(),
                        payload=payload
                    )
                )

            if current_vector_index is not None and current_vector_index < len(txt_emb):
                qdrant_dense_batch.append(
                    PointStruct(
                        id=current_vector_index,
                        vector=txt_emb[current_vector_index].tolist(),
                        payload=payload
                    )
                )

            for sec_name in ["intro", "method", "conclusion"]:
                sec_text = sections.get(sec_name, "")
                sec_text = sec_text.strip() if sec_text else ""

                if sec_text:
                    chunk_uuid = make_uuid(f"rag_{pid}_{sec_name}")

                    rag_ids.append(chunk_uuid)
                    rag_texts.append(sec_text)
                    rag_payloads.append({
                        "paper_id": pid,
                        "section": sec_name,
                        "title": title,
                        "year": year
                    })

            core_text = ". ".join(
                text.strip()
                for text in (title, abstract)
                if text.strip()
            )

            if core_text:
                multivec_ids.append(make_uuid(f"multivec_{pid}_core"))
                multivec_texts.append(core_text)
                multivec_payloads.append({
                    "paper_id": pid,
                    "section": "core",
                    "title": title,
                    "year": year
                })

            for sec_name in ["intro", "method", "conclusion"]:
                sec_text = sections.get(sec_name, "")
                sec_text = sec_text.strip() if sec_text else ""

                if sec_text:
                    multivec_ids.append(make_uuid(f"multivec_{pid}_{sec_name}"))
                    multivec_texts.append(sec_text)
                    multivec_payloads.append({
                        "paper_id": pid,
                        "section": sec_name,
                        "title": title,
                        "year": year
                    })

            if (
                len(qdrant_dense_batch) >= BATCH_SIZE
                or len(rag_texts) >= BATCH_SIZE * 3
                or len(multivec_texts) >= BATCH_SIZE * 4
            ):
                flush_batches(
                    qdrant=qdrant,
                    nlp_model=nlp_model,
                    qdrant_gcn_batch=qdrant_gcn_batch,
                    qdrant_dense_batch=qdrant_dense_batch,
                    rag_ids=rag_ids,
                    rag_texts=rag_texts,
                    rag_payloads=rag_payloads,
                    multivec_ids=multivec_ids,
                    multivec_texts=multivec_texts,
                    multivec_payloads=multivec_payloads
                )

                qdrant_gcn_batch.clear()
                qdrant_dense_batch.clear()

                rag_ids.clear()
                rag_texts.clear()
                rag_payloads.clear()

                multivec_ids.clear()
                multivec_texts.clear()
                multivec_payloads.clear()

    if (
        qdrant_gcn_batch
        or qdrant_dense_batch
        or rag_texts
        or multivec_texts
    ):
        flush_batches(
            qdrant=qdrant,
            nlp_model=nlp_model,
            qdrant_gcn_batch=qdrant_gcn_batch,
            qdrant_dense_batch=qdrant_dense_batch,
            rag_ids=rag_ids,
            rag_texts=rag_texts,
            rag_payloads=rag_payloads,
            multivec_ids=multivec_ids,
            multivec_texts=multivec_texts,
            multivec_payloads=multivec_payloads
        )

    print("\n QDRANT hoàn thành")
    print("F1_v2 đã có dữ liệu trong collection papers_multivec.")


def flush_batches(
    qdrant,
    nlp_model,
    qdrant_gcn_batch,
    qdrant_dense_batch,
    rag_ids,
    rag_texts,
    rag_payloads,
    multivec_ids,
    multivec_texts,
    multivec_payloads
):
    if qdrant_gcn_batch:
        qdrant.upsert(
            collection_name="gcn_collection",
            points=qdrant_gcn_batch
        )

    if qdrant_dense_batch:
        qdrant.upsert(
            collection_name="dense_collection",
            points=qdrant_dense_batch
        )

    if rag_texts:
        rag_vectors = nlp_model.encode(
            rag_texts,
            batch_size=256,
            show_progress_bar=False
        )

        rag_points = [
            PointStruct(
                id=r_id,
                vector=vec.tolist(),
                payload=payload
            )
            for r_id, vec, payload in zip(
                rag_ids,
                rag_vectors,
                rag_payloads
            )
        ]

        qdrant.upsert(
            collection_name="rag_chunk_collection",
            points=rag_points
        )

    if multivec_texts:
        multivec_vectors = nlp_model.encode(
            multivec_texts,
            batch_size=256,
            show_progress_bar=False
        )

        multivec_points = [
            PointStruct(
                id=m_id,
                vector=vec.tolist(),
                payload=payload
            )
            for m_id, vec, payload in zip(
                multivec_ids,
                multivec_vectors,
                multivec_payloads
            )
        ]

        qdrant.upsert(
            collection_name="papers_multivec",
            points=multivec_points
        )
if __name__ == "__main__":
    main()
