import json
from pathlib import Path
import psycopg2
from psycopg2.extras import execute_values
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_FILE = BASE_DIR / "data/02_processed/papers_100k_complete.jsonl"
BATCH_SIZE = 1000

def normalize_authors(raw_authors):
    """
    Luu tac gia kem ID that tu Semantic Scholar neu co.
    Neu paper thieu authorId thi fallback ve name de khong mat tac gia.
    """
    authors = []
    for author in raw_authors or []:
        if isinstance(author, dict):
            name = (author.get("name") or "").strip()
            author_id = author.get("authorId") or author.get("author_id") or author.get("id")
            if author_id or name:
                authors.append({
                    "author_id": str(author_id or name),
                    "name": name or str(author_id)
                })
        elif isinstance(author, str) and author.strip():
            authors.append({
                "author_id": author.strip(),
                "name": author.strip()
            })
    return authors

def init_db(cursor, conn):
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS papers (
            paper_id VARCHAR(50) PRIMARY KEY,
            title TEXT,
            year INT,
            abstract TEXT,
            intro_text TEXT,
            method_text TEXT,
            conclusion_text TEXT,
            authors JSONB
        );
    """)
    conn.commit()
    print("  [+] Bảng 'papers' đã sẵn sàng.")

def main():
    print("BẮT ĐẦU MIGRATION VÀO POSTGRESQL...")
    
    try:
        conn = psycopg2.connect("dbname=paper_recommender user=postgresql password=postgresql host=localhost")
        cursor = conn.cursor()
        init_db(cursor, conn)
    except Exception as e:
        print(f"Lỗi kết nối Postgres: {e}")
        return

    pg_batch = []
    
    print("[*] Đang đếm số lượng bài báo...")
    total_lines = sum(1 for _ in open(INPUT_FILE, 'r', encoding='utf-8'))
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in tqdm(f, total=total_lines, desc="Đang bơm Data vào Postgres"):
            if not line.strip(): continue
            p = json.loads(line)
            
            raw_pid = p.get("corpusId") or p.get("corpusid") or p.get("paperId")
            if not raw_pid:
                continue
            pid = str(raw_pid)
            authors = normalize_authors(p.get("authors", []))
            sections = p.get("sections", {}) or {}

            pg_batch.append((
                pid, 
                p.get("title", "") or "", 
                p.get("year"), 
                p.get("abstract", ""), 
                sections.get("intro", ""), 
                sections.get("method", ""), 
                sections.get("conclusion", ""), 
                json.dumps(authors)
            ))

            if len(pg_batch) >= BATCH_SIZE:
                execute_values(cursor, """
                    INSERT INTO papers (paper_id, title, year, abstract, intro_text, method_text, conclusion_text, authors)
                    VALUES %s
                    ON CONFLICT (paper_id) DO UPDATE SET
                        title = EXCLUDED.title,
                        year = EXCLUDED.year,
                        abstract = EXCLUDED.abstract,
                        intro_text = EXCLUDED.intro_text,
                        method_text = EXCLUDED.method_text,
                        conclusion_text = EXCLUDED.conclusion_text,
                        authors = EXCLUDED.authors
                """, pg_batch)
                conn.commit()
                pg_batch.clear()

    if pg_batch:
        execute_values(cursor, """
            INSERT INTO papers (paper_id, title, year, abstract, intro_text, method_text, conclusion_text, authors)
            VALUES %s
            ON CONFLICT (paper_id) DO UPDATE SET
                title = EXCLUDED.title,
                year = EXCLUDED.year,
                abstract = EXCLUDED.abstract,
                intro_text = EXCLUDED.intro_text,
                method_text = EXCLUDED.method_text,
                conclusion_text = EXCLUDED.conclusion_text,
                authors = EXCLUDED.authors
        """, pg_batch)
        conn.commit()

    print("\nĐÃ NẠP ĐẦY DỮ LIỆU VÀO PSQL")
    cursor.close(); conn.close()

if __name__ == "__main__":
    main()
