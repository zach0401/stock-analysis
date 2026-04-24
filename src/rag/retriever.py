from src.rag.ingestor import load_existing_vectorstore
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document
import os

CHROMA_DIR = "./data/chroma_db"

# Reranker model — scores query+chunk pairs for precision
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_reranker = None # lazy load


def hybrid_search(vectorstore, query: str, k: int = 10) -> list:
    """
    Combine vector search + BM25 keyword search.
    Returns top-k candidates for reranking.
    """

    # --- Vector search (semantic) ---
    vector_results = vectorstore.similarity_search(query, k=k)

    # --- BM25 search (keyword) ---
    # Get all documents from the collection for BM25
    all_docs = vectorstore.get()
    all_texts = all_docs.get("documents", [])
    all_metadatas = all_docs.get("metadatas", [])

    if not all_texts:
        return vector_results

    # Tokenize for BM25
    tokenized_corpus = [text.lower().split() for text in all_texts]
    bm25 = BM25Okapi(tokenized_corpus)

    # Get BM25 scores
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)

    # Get top-k BM25 results
    top_bm25_indices = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True
    )[:k]

    bm25_results = [
        Document(
            page_content=all_texts[i],
            metadata=all_metadatas[i] if i < len(all_metadatas) else {}
        )
        for i in top_bm25_indices
    ]

    # --- Combine and deduplicate ---
    seen = set()
    combined = []

    for doc in vector_results + bm25_results:
        # Use first 100 chars as dedup key
        key = doc.page_content[:100]
        if key not in seen:
            seen.add(key)
            combined.append(doc)

    return combined


RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_reranker = None  # lazy loaded

def rerank(query: str, candidates: list, top_k: int = 4) -> list:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANKER_MODEL)
    
    if not candidates:
        return []

    pairs = [[query, doc.page_content] for doc in candidates]
    scores = _reranker.predict(pairs)
    scored = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:top_k]]


def get_relevant_context(query: str, ticker: str = "stock_docs", k: int = 4) -> str:
    """
    Full retrieval pipeline:
    1. Hybrid search (vector + BM25) → 10 candidates
    2. Cross-encoder reranking → top 4
    3. Format as string for LLM prompt
    """

    if not os.path.exists(CHROMA_DIR):
        return "No documents have been ingested yet."

    try:
        vectorstore = load_existing_vectorstore(collection_name=ticker)

        # Step 1 — hybrid retrieval (10 candidates)
        candidates = hybrid_search(vectorstore, query, k=10)

        if not candidates:
            return f"No relevant context found for {ticker}."

        # Step 2 — rerank to top 4
        top_chunks = rerank(query, candidates, top_k=k)

        # Step 3 — format for LLM
        context_parts = []
        for i, doc in enumerate(top_chunks):
            section = doc.metadata.get("section", "Unknown")
            page = doc.metadata.get("page", "?")
            context_parts.append(
                f"[Chunk {i+1} | Ticker: {ticker} | Section: {section} | Page {page}]\n{doc.page_content}"
            )

        return "\n\n---\n\n".join(context_parts)

    except Exception as e:
        return f"Could not retrieve context for {ticker}: {str(e)}"


if __name__ == "__main__":
    test_cases = [
        ("What are the main risk factors?", "AAPL"),
        ("What was the total net sales revenue?", "AAPL"),
        ("What does the company say about competition?", "AAPL"),
    ]

    for query, ticker in test_cases:
        print(f"\nQuery: {query} | Ticker: {ticker}")
        print("-" * 50)
        context = get_relevant_context(query, ticker=ticker, k=2)
        print(context[:500])
        print("...")