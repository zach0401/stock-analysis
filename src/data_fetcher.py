import yfinance as yf
import pandas as pd


def format_large_number(value) -> str:
    """Convert raw large numbers to readable format."""
    if value == "N/A" or value is None:
        return "N/A"
    try:
        value = float(value)
        if value >= 1_000_000_000_000:
            return f"${value / 1_000_000_000_000:.2f}T"
        elif value >= 1_000_000_000:
            return f"${value / 1_000_000_000:.2f}B"
        elif value >= 1_000_000:
            return f"${value / 1_000_000:.2f}M"
        else:
            return f"${value:,.0f}"
    except:
        return str(value)


def get_stock_info(ticker: str) -> dict:
    stock = yf.Ticker(ticker)
    info = stock.info

    return {
        "ticker": ticker.upper(),
        "company_name": info.get("longName", "N/A"),
        "sector": info.get("sector", "N/A"),
        "industry": info.get("industry", "N/A"),
        "current_price": info.get("currentPrice", "N/A"),
        "market_cap": format_large_number(info.get("marketCap")),
        "revenue": format_large_number(info.get("totalRevenue")),
        "net_income": format_large_number(info.get("netIncomeToCommon")),
        "pe_ratio": round(info.get("trailingPE", 0), 2) if info.get("trailingPE") else "N/A",
        "forward_pe": round(info.get("forwardPE", 0), 2) if info.get("forwardPE") else "N/A",
        "52w_high": info.get("fiftyTwoWeekHigh", "N/A"),
        "52w_low": info.get("fiftyTwoWeekLow", "N/A"),
        "profit_margin": f"{round(info.get('profitMargins', 0) * 100, 1)}%" if info.get("profitMargins") else "N/A",
        "debt_to_equity": info.get("debtToEquity", "N/A"),
        "analyst_target": info.get("targetMeanPrice", "N/A"),
        "recommendation": info.get("recommendationKey", "N/A"),
        "business_summary": info.get("longBusinessSummary", "N/A"),
    }


def get_price_history(ticker: str, period: str = "6mo") -> pd.DataFrame:
    """Fetch historical price data. Period options: 1mo, 3mo, 6mo, 1y, 2y"""
    stock = yf.Ticker(ticker)
    history = stock.history(period=period)
    return history


def get_recent_news(ticker: str) -> list[dict]:
    """Fetch recent news headlines for a ticker."""
    stock = yf.Ticker(ticker)
    news = stock.news[:5]  # top 5 articles

    cleaned = []
    for article in news:
        cleaned.append({
            "title": article.get("title", ""),
            "publisher": article.get("publisher", ""),
            "link": article.get("link", ""),
        })
    return cleaned


if __name__ == "__main__":
    # Quick test — run this file directly to check it works
    info = get_stock_info("AAPL")
    for key, value in info.items():
        print(f"{key}: {value}")

    print("\n--- Recent News ---")
    news = get_recent_news("AAPL")
    for article in news:
        print(f"  {article['title']} ({article['publisher']})")