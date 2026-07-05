import json
from pathlib import Path


def log(message: str) -> None:
    print(message, flush=True)


def load_benchmark(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        benchmark = json.load(file)
    for item in benchmark:
        item["judgments"] = {
            str(paper_id): int(grade)
            for paper_id, grade in item["judgments"].items()
        }
    return benchmark


def get_benchmark_connections() -> dict | None:
    try:
        import psycopg2
        import torch
        from qdrant_client import QdrantClient
        from sentence_transformers import SentenceTransformer
    except ModuleNotFoundError as exc:
        log(f"Thiếu thư viện: {exc.name}")
        return None

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        log(f"Đang tải SentenceTransformer trên {device}")
        if device == "cuda":
            log(f"[khởi tạo] GPU: {torch.cuda.get_device_name(0)}")
        return {
            "pg": psycopg2.connect(
                dbname="paper_recommender",
                user="postgresql",
                password="postgresql",
                host="localhost",
            ),
            "qdrant": QdrantClient(url="http://localhost:6333"),
            "nlp_model": SentenceTransformer("all-MiniLM-L6-v2", device=device),
        }
    except Exception as exc:
        log(f"Ko ket noi dc: {exc}")
        return None
