from src.data_fetcher import get_stock_info, get_recent_news
from src.rag.retriever import get_relevant_context
import time
import json


def tool_fetch_market_data(ticker: str) -> dict:
    """
    Tool 1: Fetch live market data for a ticker.
    Returns stock info and recent news as a dict.
    """
    start = time.time()

    stock_data = get_stock_info(ticker)
    news = get_recent_news(ticker)

    result = {
        "stock_data": stock_data,
        "news": news,
        "latency_seconds": round(time.time() - start, 2)
    }

    print(f"[Tool] fetch_market_data({ticker}) — {result['latency_seconds']}s")
    return result


def tool_retrieve_filing_context(ticker: str, question: str) -> dict:
    """
    Tool 2: Retrieve relevant chunks from ingested documents.
    Scoped to the specific ticker's ChromaDB collection.
    """
    start = time.time()

    context = get_relevant_context(query=question, ticker=ticker, k=4)
    latency = round(time.time() - start, 2)

    print(f"[Tool] retrieve_filing_context({ticker}) — {latency}s")

    return {
        "context": context,
        "ticker": ticker,
        "question": question,
        "latency_seconds": latency
    }


def tool_summarize_with_citations(
    stock_data: dict,
    news: list,
    rag_context: str,
    ticker: str,
    question: str = None        # add this
) -> dict:
    from src.llm_engine import analyze_stock
    start = time.time()

    raw_analysis = analyze_stock(stock_data, news, rag_context, question=question)  # pass it here
    latency = round(time.time() - start, 2)

    print(f"[Tool] summarize_with_citations({ticker}) — {latency}s")

    return {
        "analysis": raw_analysis,
        "ticker": ticker,
        "latency_seconds": latency
    }