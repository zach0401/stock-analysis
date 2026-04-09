import streamlit as st
from src.agent import run_agent
from src.rag.ingestor import ingest_document
import os
import tempfile

st.set_page_config(
    page_title="Stock Analysis Agent",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Stock Analysis Agent")
st.caption("Powered by Ollama + RAG — live market data + SEC filing context")

# --- Sidebar: Document ingestion ---
with st.sidebar:
    st.header("📄 Ingest Documents")
    st.caption("Upload a 10-K PDF to enable RAG for that ticker")

    uploaded_file = st.file_uploader("Upload 10-K PDF", type=["pdf"])
    doc_ticker = st.text_input("Ticker for this document (e.g. AAPL)").upper()

    if st.button("Ingest Document") and uploaded_file and doc_ticker:
        with st.spinner(f"Ingesting {uploaded_file.name}..."):
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            try:
                ingest_document(tmp_path, collection_name=doc_ticker)
                st.success(f"Ingested {uploaded_file.name} as {doc_ticker}")
            except Exception as e:
                st.error(f"Ingestion failed: {e}")
            finally:
                os.unlink(tmp_path)

    st.divider()
    st.header("ℹ️ How it works")
    st.markdown("""
    1. Enter a ticker symbol
    2. Ask a specific question or run full analysis
    3. Agent fetches live market data
    4. Agent searches ingested documents for context
    5. LLM synthesizes everything into structured analysis
    """)

# --- Main: Analysis ---
col1, col2 = st.columns([1, 2])

with col1:
    ticker_input = st.text_input(
        "Ticker Symbol",
        placeholder="AAPL",
        max_chars=10
    ).upper()

    question_input = st.text_area(
        "Specific Question (optional)",
        placeholder="What are the main risk factors?",
        height=100
    )

    analyze_button = st.button("Run Analysis", type="primary", use_container_width=True)

# --- Results ---
with col2:
    if analyze_button and ticker_input:
        with st.spinner(f"Agent analyzing {ticker_input}..."):
            result = run_agent(
                ticker=ticker_input,
                question=question_input if question_input else None
            )

        if "error" in result:
            st.error(result["error"])
        else:
            # Header metrics
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Company", result["company_name"][:15])
            m2.metric("Price", f"${result['current_price']}")
            m3.metric("Sector", result["sector"][:12])
            m4.metric("RAG Active", "Yes" if result["rag_used"] else "No")

            st.divider()

            # Analysis output
            st.subheader("Analysis")
            st.markdown(result["analysis"])

            st.divider()

            # Observability — session log
            with st.expander("🔍 Session Log (observability)"):
                for entry in result["log"]:
                    tool = entry["tool"]
                    latency = entry.get("latency", "?")
                    status = entry.get("status", "")
                    has_ctx = entry.get("has_context", "")

                    if tool == "retrieve_filing_context":
                        st.write(f"✅ `{tool}` — {latency}s — context found: {has_ctx}")
                    else:
                        st.write(f"✅ `{tool}` — {latency}s — {status}")

                st.write(f"**Total latency:** {result['total_latency_seconds']}s")

    elif analyze_button and not ticker_input:
        st.warning("Please enter a ticker symbol.")
    else:
        st.info("Enter a ticker symbol and click Run Analysis to begin.")