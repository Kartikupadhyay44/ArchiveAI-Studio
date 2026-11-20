"""
ArchiveAI — Modern History & Archival Chronicle Assistant
Streamlit Interface: Clean, Apple-inspired Academic Research Dashboard
"""

import os
import json
from pathlib import Path
import streamlit as st

from ingest import (
    is_index_ready,
    load_manifest,
    run_ingestion,
    resolve_pdf_path,
    CHROMA_DIR,
    COLLECTION_NAME,
)
from rag import answer, retrieve_chunks

# Page configuration
st.set_page_config(
    page_title="ArchiveAI — Indian Railways & Telegraph History",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Apple/Linear minimalist design
st.markdown(
    """
    <style>
    /* Global Background and Typography */
    .stApp {
        background-color: #fbfbfd;
        color: #1d1d1f;
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    /* Top Title Styles */
    .archive-header {
        margin-top: -1.5rem;
        margin-bottom: 0.5rem;
    }
    .archive-title {
        font-size: 2.25rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        color: #111827;
        margin-bottom: 0.25rem;
    }
    .archive-subtitle {
        font-size: 1.15rem;
        font-weight: 500;
        color: #0071e3;
        margin-bottom: 0.5rem;
    }
    .archive-desc {
        font-size: 0.95rem;
        color: #6b7280;
        max-width: 800px;
        line-height: 1.5;
        margin-bottom: 1.5rem;
    }

    /* Minimalist Cards */
    .status-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .status-card:hover {
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.04);
    }
    .status-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.05em;
        color: #9ca3af;
        margin-bottom: 0.25rem;
    }
    .status-value {
        font-size: 1.15rem;
        font-weight: 600;
        color: #1f2937;
    }
    .status-badge-ready {
        display: inline-block;
        background-color: #ecfdf5;
        color: #065f46;
        padding: 0.15rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .status-badge-not-ready {
        display: inline-block;
        background-color: #fef2f2;
        color: #991b1b;
        padding: 0.15rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
    }

    /* Answer Cards */
    .answer-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-left: 4px solid #0071e3;
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 1.25rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.03);
    }
    .answer-card-notfound {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-left: 4px solid #f59e0b;
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 1.25rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.03);
    }
    .answer-title {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #6b7280;
        margin-bottom: 0.5rem;
    }
    .answer-content {
        font-size: 1.25rem;
        font-weight: 600;
        color: #111827;
        line-height: 1.45;
        margin-bottom: 1rem;
    }
    .source-pill {
        display: inline-flex;
        align-items: center;
        background: #eff6ff;
        color: #1d4ed8;
        border: 1px solid #bfdbfe;
        border-radius: 6px;
        padding: 0.2rem 0.65rem;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 0.5rem;
    }
    .not-found-pill {
        display: inline-flex;
        align-items: center;
        background: #fef3c7;
        color: #92400e;
        border: 1px solid #fde68a;
        border-radius: 6px;
        padding: 0.2rem 0.65rem;
        font-size: 0.85rem;
        font-weight: 600;
    }

    /* Assessment block */
    .assessment-box {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar
with st.sidebar:
    st.markdown("### 🏛️ ArchiveAI")
    st.caption("Generative AI Mid-Term · Project 9")
    st.markdown("---")

    # Document details
    st.markdown("**Assigned Knowledge Source:**")
    st.code("indian_railways_and_telegraph_history_50p.pdf", language="text")

    # Dynamic status
    chroma_path = Path(CHROMA_DIR)
    is_ready = False
    manifest = load_manifest(chroma_path) if chroma_path.is_dir() else None
    
    try:
        pdf_path = resolve_pdf_path()
        is_ready = is_index_ready(pdf_path, chroma_path, COLLECTION_NAME)
    except Exception:
        is_ready = False

    st.markdown("**Vector Store Status:**")
    if is_ready:
        st.success(f"Index Ready ({manifest.get('num_chunks', 0)} Chunks)")
    else:
        st.warning("Not Initialized")

    if st.button("🔄 Rebuild Knowledge Base", use_container_width=True):
        with st.spinner("Re-extracting PDF, generating embeddings, and updating ChromaDB..."):
            try:
                res = run_ingestion(force_rebuild=True)
                st.success(f"Indexed {res['num_pages']} pages into {res['num_chunks']} chunks!")
                st.rerun()
            except Exception as e:
                st.error(f"Ingestion failed: {e}")

    st.markdown("---")
    st.markdown("### 🛡️ Grounding Safeguards")
    st.markdown(
        """
        - **PDF-Only Context**: Answers strictly derived from archival pages.
        - **Local MiniLM-L6-v2**: Zero API cost, private local embeddings.
        - **Strict Hallucination Gate**: Returns `[not found]` when evidence is absent.
        - **Deterministic Fallback**: Rigorous factual verification when Ollama is offline.
        """
    )
    st.caption("Engineered for Apple Silicon · 100% Offline Compatible")

# Main Header
st.markdown('<div class="archive-header">', unsafe_allow_html=True)
st.markdown('<div class="archive-title">ArchiveAI</div>', unsafe_allow_html=True)
st.markdown('<div class="archive-subtitle">Explore the History of Indian Railways & Telegraphs</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="archive-desc">'
    'An AI-powered historical research assistant grounded in a 50-page archival document. '
    'Every answer is cited directly from authentic colonial dispatches and records, '
    'with strict protection against factual hallucination.'
    '</div>',
    unsafe_allow_html=True,
)
st.markdown('</div>', unsafe_allow_html=True)

# Document Status Cards
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
        <div class="status-card">
            <div class="status-label">Document Corpus</div>
            <div class="status-value">50 Pages Archival PDF</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        """
        <div class="status-card">
            <div class="status-label">Knowledge Base</div>
            <div class="status-value">ChromaDB Persistent</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    status_badge = (
        f'<span class="status-badge-ready">● Ready ({manifest.get("num_chunks", 97)} chunks)</span>'
        if is_ready
        else '<span class="status-badge-not-ready">● Not Initialized</span>'
    )
    st.markdown(
        f"""
        <div class="status-card">
            <div class="status-label">System State</div>
            <div class="status-value">{status_badge}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# Main Query Section
st.subheader("🔍 Archival Query Search")
user_query = st.text_input(
    label="Search query",
    placeholder="Ask anything about the historical document (e.g. 1849 guarantee system, Bhor Ghat incline, O'Shaughnessy telegraph)...",
    label_visibility="collapsed",
    key="main_search_input",
)

search_col1, search_col2 = st.columns([1, 5])
with search_col1:
    search_clicked = st.button("Search Archive", type="primary", use_container_width=True)

if search_clicked or (user_query and st.session_state.get("submitted_query") != user_query):
    if not user_query.strip():
        st.warning("Please enter a question to query the archive.")
    else:
        st.session_state["submitted_query"] = user_query
        with st.spinner("Retrieving verified archival evidence..."):
            result = answer(user_query)
            retrieved = retrieve_chunks(user_query, top_k=2)

        ans_str = result.get("answer", "[not found]")
        sources = result.get("sources", [])

        if ans_str == "[not found]":
            st.markdown(
                """
                <div class="answer-card-notfound">
                    <div class="answer-title">Evidence Search Result</div>
                    <div class="answer-content"><code>[not found]</code></div>
                    <div>
                        <span class="not-found-pill">Status: Unsupported Question</span>
                        <span style="font-size:0.9rem; color:#6b7280; margin-left: 0.5rem;">
                            The provided 50-page historical corpus does not contain verified records answering this query.
                        </span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            pills = "".join([f'<span class="source-pill">📄 Source: {src.replace("_", " ").title()}</span>' for src in sources])
            st.markdown(
                f"""
                <div class="answer-card">
                    <div class="answer-title">Grounded Archival Answer</div>
                    <div class="answer-content">{ans_str}</div>
                    <div>{pills}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if retrieved:
            with st.expander("📖 View Supporting Retrieved Excerpt (Authentic PDF Text)"):
                for rank, chk in enumerate(retrieved, 1):
                    st.markdown(f"**Rank {rank} — {chk['page_label'].replace('_', ' ').title()}** *(Distance: {chk['distance']:.4f})*")
                    st.info(chk["text"])

st.markdown("<hr style='margin: 2.5rem 0 2rem 0; border: none; border-top: 1px solid #e5e7eb;'>", unsafe_allow_html=True)

# Official Assessment Section
st.subheader("📋 Official Assessment Questions")
st.caption(
    "Execute the 5 assigned mid-term examination questions through the real RAG pipeline. "
    "Results display genuine extracted answers, verified page numbers, and anti-hallucination status."
)

OFFICIAL_QUESTIONS = [
    {
        "id": "Q1",
        "question": (
            "What exact annual guaranteed interest rate, expressed as a percentage, "
            "did the colonial administration promise private British railway companies "
            "under the 1849 contract terms?"
        ),
        "expected": "5% per annum (Page 13)",
    },
    {
        "id": "Q2",
        "question": (
            "Which engineering obstacle on the Bhor Ghat incline made the reversing "
            "station described in Chapter 3 necessary, and what was the gradient ratio?"
        ),
        "expected": "The sheer basalt precipice of Battery Hill above the Khandala ravine; 1 in 37 (Page 26)",
    },
    {
        "id": "Q3",
        "question": (
            "What is the ticket price for an Executive Chair Car seat on the "
            "Vande Bharat Express between New Delhi and Varanasi?"
        ),
        "expected": "[not found] (Unsupported contemporary question)",
    },
    {
        "id": "Q4",
        "question": (
            "According to the archival records in the text, who supervised the laying of "
            "the experimental Diamond Harbour underwater telegraph cable in 1851?"
        ),
        "expected": "Dr. William Brooke O'Shaughnessy, Assay Master of the Calcutta Mint (Page 43)",
    },
    {
        "id": "Q5",
        "question": "Which country won the 2022 FIFA World Cup?",
        "expected": "[not found] (Unsupported contemporary question)",
    },
]

# Run All button
if st.button("🚀 Run All 5 Assessment Questions", type="primary"):
    st.session_state["run_all_active"] = True

run_all = st.session_state.get("run_all_active", False)

for item in OFFICIAL_QUESTIONS:
    q_id = item["id"]
    q_text = item["question"]

    with st.container():
        q_col, btn_col = st.columns([5, 1])
        with q_col:
            st.markdown(f"**{q_id}:** {q_text}")
        with btn_col:
            btn_run = st.button(f"Run {q_id}", key=f"btn_{q_id}")

        if run_all or btn_run or st.session_state.get(f"active_{q_id}"):
            st.session_state[f"active_{q_id}"] = True
            with st.spinner(f"Evaluating {q_id}..."):
                res = answer(q_text)
                actual_ans = res.get("answer", "[not found]")
                actual_sources = res.get("sources", [])

            if actual_ans == "[not found]":
                st.markdown(
                    f"""
                    <div style="background:#fefce8; border-left: 4px solid #f59e0b; padding: 0.75rem 1rem; border-radius: 6px; margin: 0.5rem 0 1rem 0;">
                        <span style="font-weight:600; color:#92400e;">Answer:</span> <code>[not found]</code><br>
                        <span style="font-size:0.85rem; color:#78350f;">Sources: <code>[]</code> &bull; Status: Verified Not Found</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                src_display = ", ".join([s.replace("_", " ").title() for s in actual_sources])
                st.markdown(
                    f"""
                    <div style="background:#f0fdf4; border-left: 4px solid #10b981; padding: 0.75rem 1rem; border-radius: 6px; margin: 0.5rem 0 1rem 0;">
                        <span style="font-weight:600; color:#065f46;">Answer:</span> <b>{actual_ans}</b><br>
                        <span style="font-size:0.85rem; color:#047857;">Source: <b>{src_display}</b> &bull; Status: Grounded in PDF</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
