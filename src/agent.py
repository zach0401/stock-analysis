from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from src.data_fetcher import get_stock_info, get_recent_news
from src.rag.retriever import get_relevant_context
import time


@tool
def fetch_market_data(ticker: str) -> str:
    """
    Use this to get live market data for a stock ticker.
    Returns current price, PE ratio, revenue, margins, analyst targets.
    Use when the user asks about current price, valuation, or financial metrics.
    Input: ticker symbol like AAPL, TSLA, MSFT
    """
    stock = get_stock_info(ticker)
    news = get_recent_news(ticker)
    news_text = "\n".join([f"- {n['title']}" for n in news if n.get('title')])
    if not news_text:
        news_text = "No recent news available"
    return f"MARKET DATA:\n{stock}\n\nRECENT NEWS:\n{news_text}"


@tool
def retrieve_filing_context(ticker_and_question: str) -> str:
    """
    Use this to search SEC 10-K filings for a company.
    Returns relevant excerpts with page citations.
    Use when the user asks about risks, strategy, competition, products,
    legal proceedings, R&D, or anything requiring document context.
    Input format: 'TICKER: question' e.g. 'AAPL: What are the main risks?'
    """
    if ":" in ticker_and_question:
        ticker, question = ticker_and_question.split(":", 1)
        ticker = ticker.strip().upper()
        question = question.strip()
    else:
        return "Invalid input. Use format: 'TICKER: question'"

    return get_relevant_context(query=question, ticker=ticker, k=4)


def build_agent():
    llm = ChatOllama(model="llama3.1:8b", temperature=0.3)
    tools = [fetch_market_data, retrieve_filing_context]

    return create_agent(
        model=llm,
        tools=tools,
        system_prompt="""You are an expert stock analyst with access to two tools:
1. fetch_market_data — gets live price, PE ratio, revenue, margins for any ticker
2. retrieve_filing_context — searches SEC 10-K filings for document context

RULES:
- Only call tools you actually need for the question asked
- For price/valuation questions: use fetch_market_data only
- For strategy/risk/competition questions: use retrieve_filing_context only
- For comprehensive analysis: use both tools
- Always cite page numbers when using filing context
- Never invent numbers not returned by the tools
- After receiving tool results, you MUST provide a detailed answer using the data returned
- Your response must include the actual numbers and analysis from the tool output
- Only add 'DISCLAIMER: This is not financial advice.' at the very end after your full answer"""
    )


def run_agent(ticker: str, question: str = None) -> dict:
    start = time.time()
    agent = build_agent()

    if question:
        user_input = f"Ticker: {ticker}. Question: {question}"
    else:
        user_input = f"Give me a full analysis of {ticker} including current valuation, key financial signals, main risks from their filings, and overall outlook."

    print(f"\n[Agent] Starting for {ticker}")

    try:
        result = agent.invoke({
            "messages": [{"role": "user", "content": user_input}]
        })

        analysis = result["messages"][-1].content

        # Accurately detect if RAG tool was actually called
        rag_used = any(
            hasattr(msg, "name") and msg.name == "retrieve_filing_context"
            for msg in result["messages"]
        )

        log = [{"status": "success"}]

    except Exception as e:
        analysis = f"Agent error: {str(e)}"
        rag_used = False
        log = [{"error": str(e)}]

    stock_data = get_stock_info(ticker)

    return {
        "ticker": ticker,
        "company_name": stock_data.get("company_name", ticker),
        "current_price": stock_data.get("current_price", "N/A"),
        "sector": stock_data.get("sector", "N/A"),
        "analysis": analysis,
        "rag_used": rag_used,
        "total_latency_seconds": round(time.time() - start, 2),
        "log": log
    }


if __name__ == "__main__":
    result = run_agent("AAPL", "What is the current stock price?")
    print("\n" + "=" * 60)
    print(f"Analysis: {result['analysis']}")
    print(f"RAG used: {result['rag_used']}")
    print(f"Latency: {result['total_latency_seconds']}s")