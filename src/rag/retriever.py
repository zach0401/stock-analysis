from src.rag.ingestor import load_existing_vectorstore
import os

CHROMA_DIR = "./data/chroma_db"


def get_relevant_context(query: str, ticker: str = "stock_docs", k: int = 4) -> str:
    """
    Search the vector store for chunks relevant to the query.
    Scoped to a specific ticker's collection.
    Returns formatted string to inject into LLM prompt.
    """

    # Check if any ChromaDB exists at all
    if not os.path.exists(CHROMA_DIR):
        return "No documents have been ingested yet."

    try:
        # Load the collection specific to this ticker
        vectorstore = load_existing_vectorstore(collection_name=ticker)

        # MMR — diverse retrieval (upgraded from Day 2)
        results = vectorstore.max_marginal_relevance_search(
            query, k=k, fetch_k=20
        )

        if not results:
            return f"No relevant context found for {ticker} in documents."

        context_parts = []
        for i, doc in enumerate(results):
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "?")
            context_parts.append(
                f"[Chunk {i+1} | Ticker: {ticker} | Page {page}]\n{doc.page_content}"
            )

        return "\n\n---\n\n".join(context_parts)

    except Exception as e:
        return f"Could not retrieve context for {ticker}: {str(e)}"


if __name__ == "__main__":
    test_queries = [
        ("What are the main risk factors?", "AAPL"),
        ("What was the total revenue?", "AAPL"),
        ("What does the company say about competition?", "AAPL"),
    ]

    for query, ticker in test_queries:
        print(f"\nQuery: {query} | Ticker: {ticker}")
        print("-" * 50)
        context = get_relevant_context(query, ticker=ticker, k=2)
        print(context[:400], "...")