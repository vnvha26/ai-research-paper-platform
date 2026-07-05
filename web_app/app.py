import streamlit as st
import sys
import os

st.set_page_config(page_title="AI Paper Platform", page_icon="🔬", layout="wide")

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from services.db_connection import get_db_connections
from web_app.tabs import tab_f1_search

if "conns" not in st.session_state:
    with st.spinner("Đang kết nối Database và tải mô hình AI..."):
        conns = get_db_connections()
        if conns is None:
            st.error("Hệ thống Database đang gặp sự cố. Vui lòng kiểm tra lại Docker!")
            st.stop()
        st.session_state.conns = conns

st.title("AI Research Search Engine")
st.caption("Dữ liệu: 100.000+ bài báo (2015-2024) | Công nghệ: GCN, MiniLM, PostgreSQL, Neo4j, Qdrant")

tab_f1_search.render(st.session_state.conns)
