import json
from pathlib import Path
from neo4j import GraphDatabase
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_FILE = BASE_DIR / "data/02_processed/papers_100k_complete.jsonl"
BATCH_SIZE = 1000

def normalize_authors(raw_authors):
    """
    Tao danh sach tac gia co author_id rieng.
    Neu Semantic Scholar khong tra authorId thi fallback ve name.
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

def init_indexes(session):
    print("  [+] Đang thiết lập Index cho Neo4j...")
    session.run("CREATE CONSTRAINT paper_id_unique IF NOT EXISTS FOR (p:Paper) REQUIRE p.paper_id IS UNIQUE")
    session.run("CREATE CONSTRAINT author_id_unique IF NOT EXISTS FOR (a:Author) REQUIRE a.author_id IS UNIQUE")

def build_paper_id_to_corpus_id():
    mapping = {}
    total_lines = 0
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            total_lines += 1
            p = json.loads(line)
            paper_id = p.get("paperId")
            corpus_id = p.get("corpusId") or p.get("corpusid")
            if paper_id and corpus_id:
                mapping[str(paper_id)] = str(corpus_id)
    return mapping, total_lines

def main():
    print("BẮT ĐẦU MIGRATION VÀO NEO4J...")
    
    try:
        driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "neo4jvha2601"))
        with driver.session() as session:
            init_indexes(session)
    except Exception as e:
        print(f"Lỗi kết nối Neo4j")
        return

    neo_papers_batch = []
    neo_cites_batch = []
    
    print("[*] Đang đếm số lượng bài báo...")
    paper_id_to_corpus_id, total_lines = build_paper_id_to_corpus_id()
    mapped_cites = 0
    raw_cites = 0
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in tqdm(f, total=total_lines, desc="Đang xây Đồ thị Neo4j"):
            if not line.strip(): continue
            p = json.loads(line)
            
            raw_pid = p.get("corpusId") or p.get("corpusid") or p.get("paperId")
            if not raw_pid:
                continue
            pid = str(raw_pid)
            authors = normalize_authors(p.get("authors", []))
            out_cites = [str(c) for c in p.get("outCitations", []) if c]

            neo_papers_batch.append({"paper_id": pid, "authors": authors, "title": p.get("title", "")})
            for target_id in out_cites:
                mapped_target_id = paper_id_to_corpus_id.get(target_id)
                if mapped_target_id:
                    mapped_cites += 1
                    neo_cites_batch.append({"source": pid, "target": mapped_target_id})
                else:
                    raw_cites += 1
                    neo_cites_batch.append({"source": pid, "target": target_id})

            if len(neo_papers_batch) >= BATCH_SIZE:
                with driver.session() as session:
                    session.run("""
                        UNWIND $batch AS p
                        MERGE (paper:Paper {paper_id: p.paper_id})
                        SET paper.title = p.title
                        WITH paper, p
                        UNWIND p.authors AS author
                        MERGE (a:Author {author_id: author.author_id})
                        SET a.name = author.name
                        MERGE (a)-[:WROTE]->(paper)
                    """, batch=neo_papers_batch)
                    
                    session.run("""
                        UNWIND $batch AS c
                        MERGE (p1:Paper {paper_id: c.source})
                        MERGE (p2:Paper {paper_id: c.target})
                        MERGE (p1)-[:CITES]->(p2)
                    """, batch=neo_cites_batch)
                neo_papers_batch.clear(); neo_cites_batch.clear()

    if neo_papers_batch:
        with driver.session() as session:
            session.run("""
                UNWIND $batch AS p
                MERGE (paper:Paper {paper_id: p.paper_id})
                SET paper.title = p.title
                WITH paper, p
                UNWIND p.authors AS author
                MERGE (a:Author {author_id: author.author_id})
                SET a.name = author.name
                MERGE (a)-[:WROTE]->(paper)
            """, batch=neo_papers_batch)
            session.run("UNWIND $batch AS c MERGE (p1:Paper {paper_id: c.source}) MERGE (p2:Paper {paper_id: c.target}) MERGE (p1)-[:CITES]->(p2)", batch=neo_cites_batch)

    print("\n neo4j hoàn thành")
    print(f"\nCitation targets mapped to corpusId: {mapped_cites:,}")
    print(f"Citation targets kept as raw paperId nodes: {raw_cites:,}")
    driver.close()

if __name__ == "__main__":
    main()
