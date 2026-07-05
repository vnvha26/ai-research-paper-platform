import streamlit as st
import streamlit.components.v1 as components

from services.f1_paper_search import hybrid_search_f1


def render(connections):
    if "f1_results" not in st.session_state:
        st.session_state.f1_results = []

    if "f1_page" not in st.session_state:
        st.session_state.f1_page = 1

    if "f1_query" not in st.session_state:
        st.session_state.f1_query = ""

    if "scroll_to_top" not in st.session_state:
        st.session_state.scroll_to_top = False

    if st.session_state.scroll_to_top:
        js = """
        <script>
            var body = window.parent.document.querySelector(".main");
            if (body) {
                body.scrollTop = 0;
            }
        </script>
        """
        components.html(js, height=0)
        st.session_state.scroll_to_top = False

    st.header("🔍 Tìm kiếm Bài báo Học thuật")

    with st.form(key="f1_search_form"):
        col_input, col_btn = st.columns([4, 1])

        with col_input:
            user_query = st.text_input(
                "💡 Nhập chủ đề, ví dụ: Deep learning for medical imaging...",
                value=st.session_state.f1_query,
                key="f1_search_input_box"
            )

        with col_btn:
            st.write("")
            st.write("")
            search_clicked = st.form_submit_button(
                "🚀 Tìm Kiếm",
                use_container_width=True
            )

    if search_clicked:
        if not user_query.strip():
            st.warning("⚠️ Vui lòng nhập từ khóa!")
        else:
            st.session_state.f1_query = user_query.strip()

            with st.spinner("🤖 AI đang tìm kiếm và xếp hạng bằng F1 V2..."):
                results = hybrid_search_f1(
                    connections,
                    st.session_state.f1_query,
                    top_k_final=30
                )

                st.session_state.f1_results = results
                st.session_state.f1_page = 1
                st.session_state.scroll_to_top = True

                if not results:
                    st.warning("⚠️ Không tìm thấy bài báo nào khớp. Hãy thử từ khóa khác!")
                else:
                    st.rerun()

    if st.session_state.f1_results:
        results = st.session_state.f1_results

        total_papers = len(results)
        papers_per_page = 10
        total_pages = (total_papers - 1) // papers_per_page + 1

        st.success(f"✅ Đã tìm thấy Top {total_papers} bài báo sát nghĩa nhất!")
        st.divider()

        start_idx = (st.session_state.f1_page - 1) * papers_per_page
        end_idx = start_idx + papers_per_page
        current_page_results = results[start_idx:end_idx]

        for i, paper in enumerate(current_page_results):
            rank = start_idx + i + 1

            pid = paper.get("paper_id", "N/A")
            title = paper.get("title", "Không có tiêu đề")
            year = paper.get("year", "N/A")

            final_score = paper.get("final_score", 0)
            cosine_score = paper.get("cosine_score", 0)
            bm25_score = paper.get("bm25_score", 0)
            embedding_quality = paper.get("embedding_quality", "unknown")

            abstract_text = paper.get("abstract", "") or "Không có tóm tắt cho bài báo này."
            short_abstract = (
                abstract_text[:350] + "..."
                if len(abstract_text) > 350
                else abstract_text
            )

            detail_url = f"/paper_detail?id={pid}"

            try:
                container = st.container(border=True)
            except TypeError:
                container = st.container()

            with container:
                st.markdown(f"### {rank}. [{title}]({detail_url})")

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("📅 Năm", year)

                with col2:
                    st.metric("🎯 Final Score", f"{final_score:.2%}")

                with col3:
                    st.metric("🧠 Cosine", f"{cosine_score:.2%}")

                with col4:
                    st.metric("🔎 BM25", f"{bm25_score:.2%}")

                st.caption(
                    f"🆔 ID: `{pid}` | "
                    f"📦 Embedding: `{embedding_quality}`"
                )

                st.markdown(f"**Tóm tắt:** {short_abstract}")

                try:
                    st.link_button(
                        "📖 Đọc Full Text ➡️",
                        detail_url
                    )
                except AttributeError:
                    st.markdown(f"[📖 Đọc Full Text ➡️]({detail_url})")

        st.divider()

        col_prev, col_page, col_next = st.columns([1, 2, 1])

        with col_prev:
            if st.button("⬅️ Trang Trước", key="btn_prev_bottom"):
                if st.session_state.f1_page > 1:
                    st.session_state.f1_page -= 1
                    st.session_state.scroll_to_top = True
                    st.rerun()

        with col_page:
            st.markdown(
                f"<h4 style='text-align: center; color: #555;'>"
                f"Trang {st.session_state.f1_page} / {total_pages}"
                f"</h4>",
                unsafe_allow_html=True
            )

        with col_next:
            if st.button("Trang Sau ➡️", key="btn_next_bottom"):
                if st.session_state.f1_page < total_pages:
                    st.session_state.f1_page += 1
                    st.session_state.scroll_to_top = True
                    st.rerun()
