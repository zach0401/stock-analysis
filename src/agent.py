from src.tools import (
    tool_fetch_market_data,
    tool_retrieve_filing_context,
    tool_summarize_with_citations
)
import time
import json


def run_agent(ticker: str, question: str = None) -> dict:
    """
    Main agent loop.
    Takes a ticker and optional specific question.
    Decides which tools to call, calls them, returns structured result.
    """

    ticker = ticker.upper().strip()
    session_log = []  # observability — logs every tool call
    start_total = time.time()

    print(f"\n[Agent] Starting analysis for {ticker}")
    print(f"[Agent] Question: {question or 'Full analysis'}")

    # --- Tool call 1: Always fetch live market data ---
    print(f"\n[Agent] Calling tool: fetch_market_data")
    market_result = tool_fetch_market_data(ticker)

    session_log.append({
        "tool": "fetch_market_data",
        "ticker": ticker,
        "latency": market_result["latency_seconds"],
        "status": "success" if market_result["stock_data"] else "failed"
    })

    # Check if we got valid data back
    if not market_result["stock_data"]:
        return {
            "error": f"Could not fetch market data for {ticker}. Check the ticker symbol.",
            "ticker": ticker,
            "log": session_log
        }

    stock_data = market_result["stock_data"]
    news = market_result["news"]

    # --- Tool call 2: Try to retrieve document context ---
    print(f"\n[Agent] Calling tool: retrieve_filing_context")

    # Build a smart search query from the question or use a default
    search_query = question if question else (
        f"{stock_data['company_name']} revenue risks competition "
        f"outlook business performance"
    )

    rag_result = tool_retrieve_filing_context(ticker, search_query)
    rag_context = rag_result["context"]

    session_log.append({
        "tool": "retrieve_filing_context",
        "ticker": ticker,
        "query": search_query,
        "latency": rag_result["latency_seconds"],
        "has_context": "No documents" not in rag_context and "Could not" not in rag_context
    })

    # --- Tool call 3: Summarize everything ---
    print(f"\n[Agent] Calling tool: summarize_with_citations")

    summary_result = tool_summarize_with_citations(
        stock_data=stock_data,
        news=news,
        rag_context=rag_context,
        ticker=ticker,
        question=question
    )

    session_log.append({
        "tool": "summarize_with_citations",
        "ticker": ticker,
        "latency": summary_result["latency_seconds"],
        "status": "success"
    })

    total_latency = round(time.time() - start_total, 2)

    print(f"\n[Agent] Complete. Total time: {total_latency}s")

    # Final structured result
    return {
        "ticker": ticker,
        "company_name": stock_data.get("company_name", ticker),
        "current_price": stock_data.get("current_price", "N/A"),
        "sector": stock_data.get("sector", "N/A"),
        "analysis": summary_result["analysis"],
        "rag_used": session_log[1]["has_context"],
        "total_latency_seconds": total_latency,
        "log": session_log
    }


if __name__ == "__main__":
    import json

    result = run_agent("AAPL", "What are the main risks and revenue outlook?")

    print("\n" + "=" * 60)
    print(f"Company: {result['company_name']}")
    print(f"RAG used: {result['rag_used']}")
    print(f"Total latency: {result['total_latency_seconds']}s")
    print("\nAnalysis:")
    print(result["analysis"])
    print("=" * 60)

    # Debug — print full session log
    print("\nFull Session Log:")
    print(json.dumps(result["log"], indent=2))