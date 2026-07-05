def fetch_papers(pg_conn, paper_ids):
    cursor = pg_conn.cursor()
    placeholders = ",".join(["%s"] * len(paper_ids))
    query = f"""
        SELECT paper_id, title, abstract, intro_text, method_text, conclusion_text, authors
        FROM papers
        WHERE paper_id IN ({placeholders})
    """
    try:
        cursor.execute(query, tuple(paper_ids))
        rows = cursor.fetchall()
    except Exception as exc:
        pg_conn.rollback()
        raise RuntimeError(f"Lỗi lấy bài báo từ PostgreSQL: {exc}") from exc
    finally:
        cursor.close()

    return {
        str(row[0]): {
            "title": row[1] or "",
            "abstract": row[2] or "",
            "title_abs": ". ".join(
                text.strip()
                for text in (row[1] or "", row[2] or "")
                if text.strip()
            ),
            "intro": (row[3] or "").strip(),
            "method": (row[4] or "").strip(),
            "conclusion": (row[5] or "").strip(),
            "authors": row[6] or [],
        }
        for row in rows
    }
