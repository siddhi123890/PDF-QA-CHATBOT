import streamlit as st
from ingest import load_and_split_pdf
from vectorstore import create_vectorstore
from qa_chain import create_qa_chain
import tempfile
import os

# ── Page Config ──
st.set_page_config(
    page_title="DocuQuery",
    page_icon="📄",
    layout="centered",
)

# ── Session State Init ──
if "chain" not in st.session_state:
    st.session_state.chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None
if "page_count" not in st.session_state:
    st.session_state.page_count = 0
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0


# ═══════════════════════════════
#  SIDEBAR
# ═══════════════════════════════
with st.sidebar:
    st.header("DocuQuery")
    st.caption("PDF Question & Answer")

    st.divider()

    # ── Upload ──
    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"],
        key="pdf_uploader",
    )

    if uploaded_file is not None:
        if st.session_state.pdf_name != uploaded_file.name:
            with st.spinner("Indexing document..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                chunks = load_and_split_pdf(tmp_path)
                vectorstore = create_vectorstore(chunks)
                chain, retriever = create_qa_chain(vectorstore)

                # Count pages from chunk metadata
                pages = set()
                for chunk in chunks:
                    if "page" in chunk.metadata:
                        pages.add(chunk.metadata["page"])

                st.session_state.chain = chain
                st.session_state.pdf_name = uploaded_file.name
                st.session_state.page_count = len(pages) if pages else 0
                st.session_state.chunk_count = len(chunks)
                st.session_state.chat_history = []

                os.unlink(tmp_path)

            st.rerun()

    # ── File Info ──
    if st.session_state.pdf_name:
        st.success(f"✅ {st.session_state.pdf_name}")
        col1, col2 = st.columns(2)
        col1.metric("Pages", st.session_state.page_count)
        col2.metric("Chunks", st.session_state.chunk_count)
    else:
        st.info("No document uploaded yet.")

    st.divider()

    # ── Actions ──
    if st.button("Clear chat", use_container_width=True, disabled=len(st.session_state.chat_history) == 0):
        st.session_state.chat_history = []
        st.rerun()

    if st.button("New document", use_container_width=True, disabled=st.session_state.chain is None):
        st.session_state.chain = None
        st.session_state.pdf_name = None
        st.session_state.page_count = 0
        st.session_state.chunk_count = 0
        st.session_state.chat_history = []
        st.rerun()


# ═══════════════════════════════
#  MAIN CHAT AREA
# ═══════════════════════════════

# ── Empty state ──
if st.session_state.chain is None:
    st.markdown("")
    st.markdown("")
    col_a, col_b, col_c = st.columns([1, 2, 1])
    with col_b:
        st.markdown(
            "<h3 style='text-align:center; color:gray;'>Upload a PDF to start asking questions</h3>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='text-align:center; color:darkgray; font-size:0.9rem;'>"
            "Use the sidebar to upload a document. Your questions will be answered strictly from the PDF content."
            "</p>",
            unsafe_allow_html=True,
        )

else:
    # ── Render chat history ──
    for entry in st.session_state.chat_history:
        with st.chat_message(entry["role"]):
            st.markdown(entry["content"])

            # Show sources under assistant messages
            if entry["role"] == "assistant" and entry.get("sources"):
                with st.expander("📌 Sources"):
                    for i, src in enumerate(entry["sources"], 1):
                        page = src.get("page", "?")
                        snippet = src.get("snippet", "")
                        st.markdown(f"**Source {i}** — Page {page}")
                        st.caption(snippet)
                        if i < len(entry["sources"]):
                            st.divider()

    # ── Chat input ──
    if question := st.chat_input("Ask a question about your document..."):
        # Display user message
        with st.chat_message("user"):
            st.markdown(question)

        # Generate answer
        with st.chat_message("assistant"):
            with st.spinner("Searching document..."):
                # Use invoke_with_sources if available, otherwise fallback
                if hasattr(st.session_state.chain, "invoke_with_sources"):
                    answer, source_docs = st.session_state.chain.invoke_with_sources(question)
                else:
                    answer = st.session_state.chain.invoke(question)
                    source_docs = []

            st.markdown(answer)

            # Build source info
            sources = []
            for doc in source_docs:
                page_num = doc.metadata.get("page", "?")
                # page is 0-indexed in PyPDFLoader, display as 1-indexed
                if isinstance(page_num, int):
                    page_num = page_num + 1
                snippet = doc.page_content[:200].strip()
                if len(doc.page_content) > 200:
                    snippet += "..."
                sources.append({"page": page_num, "snippet": snippet})

            # Show sources expander
            if sources:
                with st.expander("📌 Sources"):
                    for i, src in enumerate(sources, 1):
                        st.markdown(f"**Source {i}** — Page {src['page']}")
                        st.caption(src["snippet"])
                        if i < len(sources):
                            st.divider()

        # Save to history
        st.session_state.chat_history.append({
            "role": "user",
            "content": question,
        })
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
        })
