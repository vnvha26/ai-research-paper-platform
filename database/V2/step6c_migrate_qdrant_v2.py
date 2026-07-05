import json
import torch
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parents[2]
INPUT_FILE = BASE_DIR / "data/02_processed/papers_100k_complete.jsonl"
BATCH_SIZE = 512

COLLECTION_NAME = "papers_multivec"
VECTOR_DIM      = 384
MODEL_NAME      = "all-MiniLM-L6-v2"


def init_qdrant(client: QdrantClient):
    """
    Tạo collection papers_multivec nếu chưa có.
    Mỗi point = 1 section của 1 paper, payload gồm:
      { paper_id, section, title, year }
    """
    print("  [+] Thiết lập Qdrant collection...")
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE)
        )
        print(f"  [+] Đã tạo collection '{COLLECTION_NAME}' (dim={VECTOR_DIM})")
    else:
        print(f"  [=] Collection '{COLLECTION_NAME}' đã tồn tại, bỏ qua tạo mới.")


def build_section_texts(paper: dict) -> dict[str, str]:
    """
    Trả về dict { section_name: text } cho paper.
    - core   = title + abstract (luôn có)
    - intro  = intro_text hoặc sections.intro
    - method = method_text hoặc sections.method
    - conclusion = conclusion_text hoặc sections.conclusion
    """
    title    = paper.get("title", "") or ""
    abstract = paper.get("abstract", "") or ""

    sections_dict = paper.get("sections", {}) or {}

    intro      = paper.get("intro_text")      or sections_dict.get("intro", "")      or ""
    method     = paper.get("method_text")     or sections_dict.get("method", "")     or ""
    conclusion = paper.get("conclusion_text") or sections_dict.get("conclusion", "") or ""

    result = {}

    core_text = f"{title} {abstract}".strip()
    if core_text:
        result["core"] = core_text

    if intro.strip():
        result["intro"] = intro.strip()
    if method.strip():
        result["method"] = method.strip()
    if conclusion.strip():
        result["conclusion"] = conclusion.strip()

    return result


def main():
    print("🚀 BẮT ĐẦU MIGRATION VÀO QDRANT (papers_multivec)...")

    try:
        qdrant = QdrantClient(url="http://localhost:6333")
        init_qdrant(qdrant)
    except Exception as e:
        print(f"❌ Lỗi kết nối Qdrant: {e}")
        return

    print(f"\n[*] Nạp model '{MODEL_NAME}'...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  [+] Device: {device}")
    nlp_model = SentenceTransformer(MODEL_NAME, device=device)

    print("[*] Đang đếm số bài báo...")
    total_lines = sum(1 for _ in open(INPUT_FILE, "r", encoding="utf-8"))
    print(f"  [+] Tổng: {total_lines:,} bài báo\n")

    batch_texts:    list[str]        = []
    batch_payloads: list[dict]       = []
    batch_int_ids:  list[int]        = []
    global_point_id = 0

    def flush_batch():
        """Encode + upsert toàn bộ batch hiện tại."""
        nonlocal global_point_id
        if not batch_texts:
            return

        vectors = nlp_model.encode(
            batch_texts,
            batch_size=256,
            show_progress_bar=False,
            normalize_embeddings=True,
        )

        points = [
            PointStruct(id=pid, vector=vec.tolist(), payload=pay)
            for pid, vec, pay in zip(batch_int_ids, vectors, batch_payloads)
        ]
        qdrant.upsert(collection_name=COLLECTION_NAME, points=points)

        batch_texts.clear()
        batch_payloads.clear()
        batch_int_ids.clear()

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in tqdm(f, total=total_lines, desc="Đang nhúng vectors"):
            line = line.strip()
            if not line:
                continue

            paper = json.loads(line)

            pid = str(
                paper.get("corpusId") or
                paper.get("corpusid") or
                paper.get("paperId") or
                paper.get("paper_id") or
                ""
            ).strip()
            if not pid:
                continue

            title = paper.get("title", "") or ""
            year  = paper.get("year")

            section_texts = build_section_texts(paper)

            for sec_name, sec_text in section_texts.items():
                batch_texts.append(sec_text)
                batch_payloads.append({
                    "paper_id": pid,
                    "section":  sec_name,
                    "title":    title,
                    "year":     year,
                })
                batch_int_ids.append(global_point_id)
                global_point_id += 1

            if len(batch_texts) >= BATCH_SIZE * 4:
                flush_batch()

    flush_batch()

    print(f"\n🎉 XONG! Đã upsert {global_point_id:,} points vào '{COLLECTION_NAME}'.")
    print(f"   (Trung bình {global_point_id / max(total_lines, 1):.1f} section/paper)")


if __name__ == "__main__":
    main()
