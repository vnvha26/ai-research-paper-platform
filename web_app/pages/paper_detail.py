import streamlit as st
import json
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from services.db_connection import get_db_connections
from services.f2_citation_fraud.f2_fraud_detector import analyze_paper_fraud
from services.gcn_recommender import recommend_related_papers

st.set_page_config(page_title="Chi Tiết Bài Báo", page_icon="📄", layout="wide")

query_params = st.query_params
paper_id = query_params.get("id")

if not paper_id:
    st.error("❌ Không tìm thấy ID Bài Báo trong đường dẫn!")
    st.stop()

if "conns" not in st.session_state:
    st.session_state.conns = get_db_connections()

if not st.session_state.conns:
    st.error("Không thể kết nối cơ sở dữ liệu. Vui lòng kiểm tra Docker.")
    st.stop()

pg_conn = st.session_state.conns["pg"]

def get_full_paper(pid):
    cursor = pg_conn.cursor()
    query = """
        SELECT title, year, authors, abstract, intro_text, method_text, conclusion_text 
        FROM papers WHERE paper_id = %s
    """
    try:
        cursor.execute(query, (pid,))
        row = cursor.fetchone()
    except Exception as exc:
        pg_conn.rollback()
        raise RuntimeError(f"Lỗi đọc bài báo: {exc}") from exc
    finally:
        cursor.close()
    
    if not row: return None
    
    authors_raw = row[2]
    if isinstance(authors_raw, str):
        try:
            authors_list = json.loads(authors_raw)
        except json.JSONDecodeError:
            authors_list = []
    else:
        authors_list = authors_raw if authors_raw else []
    
    return {
        "title": row[0],
        "year": row[1],
        "authors": authors_list,
        "abstract": row[3],
        "intro": row[4],
        "method": row[5],
        "conclusion": row[6]
    }

def format_author_names(authors):
    names = []
    for author in authors or []:
        if isinstance(author, dict):
            name = author.get("name") or author.get("author_id") or author.get("authorId")
        else:
            name = author
        if name:
            names.append(str(name))
    return ", ".join(names)

try:
    paper = get_full_paper(paper_id)
except RuntimeError as exc:
    st.error(str(exc))
    st.stop()

if not paper:
    st.error("❌ Bài báo này không tồn tại trong Cơ sở dữ liệu của chúng tôi.")
    st.stop()

if st.button("⬅️ Quay lại Tìm kiếm"):
    st.switch_page("app.py")

st.title(paper["title"])
st.caption(f"📅 **Năm xuất bản:** {paper['year'] if paper['year'] else 'N/A'} | 🆔 **Paper ID:** `{paper_id}`")

if paper["authors"]:
    author_str = format_author_names(paper["authors"])
    st.markdown(f"👨‍🔬 **Tác giả:** {author_str}")

st.divider()

with st.expander("Kiểm tra gian lận trích dẫn", expanded=False):
    fraud_result = analyze_paper_fraud(st.session_state.conns, paper_id)

    if "error" in fraud_result:
        st.warning(fraud_result["error"])
    else:
        fraud_score = fraud_result["fraud_score"]
        score_color = "green" if fraud_score < 20 else ("orange" if fraud_score < 50 else "red")
        label = "Ít rủi ro" if fraud_score < 20 else ("Đáng chú ý" if fraud_score < 50 else "Rủi ro cao")

        st.markdown(
            f"**Mức rủi ro:** <span style='color:{score_color}; font-weight:700'>{fraud_score:.1f}%</span> "
            f"({label})",
            unsafe_allow_html=True,
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Citation phân tích", fraud_result["total_citations"])
        c2.metric("Cờ vàng", fraud_result["yellow_flags"])
        c3.metric("Cờ đỏ", fraud_result["red_flags"])
        c4.metric("Bỏ qua ngoài dataset", fraud_result.get("skipped_missing_dataset", 0))

        skipped_count = fraud_result.get("skipped_missing_dataset", 0)
        if skipped_count:
            graph_total = fraud_result.get("graph_citation_count", fraud_result["total_citations"])
            st.caption(
                f"Neo4j có {graph_total} citation edge; chỉ phân tích "
                f"{fraud_result['total_citations']} paper có trong dataset, "
                f"bỏ qua {skipped_count} node rỗng/ngoài dataset."
            )

        suspicious_items = [item for item in fraud_result["details"] if item["risk_level"] > 0]
        if suspicious_items:
            st.write("Các trích dẫn đáng chú ý:")
            for item in suspicious_items[:5]:
                color = "red" if item["risk_level"] == 2 else "orange"
                st.markdown(
                    f"- <span style='color:{color}'>[{item['status']}]</span> "
                    f"*{item['title']}* "
                    f"(content {item.get('content_score', item.get('similarity', 0.0)) * 100:.1f}%)",
                    unsafe_allow_html=True,
                )
                st.caption(
                    f"Mutual: {'có' if item.get('is_mutual') else 'không'} | "
                    f"Tác giả chung: {item.get('shared_author_count', 0)} | "
                    f"Neighbor overlap: {item.get('neighbor_jaccard', 0.0) * 100:.1f}% | "
                    f"Common neighbors: {item.get('common_neighbor_count', 0)} | "
                    f"Local density: {item.get('local_density', 0.0) * 100:.1f}%"
                )
        else:
            st.success("Chưa phát hiện trích dẫn đáng ngờ trong các citation có dữ liệu.")

st.divider()

st.subheader("Abstract (Tóm tắt)")
st.info(paper["abstract"] if paper["abstract"] else "Không có tóm tắt.")

tab_intro, tab_method, tab_concl = st.tabs(["1. Introduction", "2. Methodology", "3. Conclusion"])

with tab_intro:
    if paper["intro"]:
        st.markdown(paper["intro"])
    else:
        st.warning("Bài báo này không trích xuất được phần Mở đầu.")

with tab_method:
    if paper["method"]:
        st.markdown(paper["method"])
    else:
        st.warning("Bài báo này không trích xuất được phần Phương pháp.")

with tab_concl:
    if paper["conclusion"]:
        st.markdown(paper["conclusion"])
    else:
        st.warning("Bài báo này không trích xuất được phần Kết luận.")

st.divider()

st.subheader("Bai bao lien quan theo GCN")
st.caption("GCN goi y cac paper co vector gan voi bai hien tai trong citation graph.")

related_papers = recommend_related_papers(st.session_state.conns, paper_id, top_k=5)

if not related_papers:
    st.info("Chua tim thay goi y GCN cho bai bao nay.")
else:
    for item in related_papers:
        related_url = f"/paper_detail?id={item['paper_id']}"
        score = item.get("gcn_score", 0.0)

        try:
            rec_container = st.container(border=True)
        except TypeError:
            rec_container = st.container()

        with rec_container:
            st.markdown(f"#### [{item['title']}]({related_url})")
            col_year, col_score = st.columns(2)
            with col_year:
                st.metric("Nam", item.get("year", "N/A"))
            with col_score:
                st.metric("GCN Score", f"{score:.2%}")
            st.caption(f"ID: `{item['paper_id']}`")
