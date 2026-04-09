from src.agent import run_agent


if __name__ == "__main__":
    ticker = input("Enter ticker symbol (e.g. AAPL, TSLA, MSFT): ").strip()
    question = input("Specific question? (press Enter to skip): ").strip()

    result = run_agent(
        ticker=ticker,
        question=question if question else None
    )

    if "error" in result:
        print(f"\nError: {result['error']}")
    else:
        print("\n" + "=" * 60)
        print(f"Company:  {result['company_name']}")
        print(f"Price:    ${result['current_price']}")
        print(f"Sector:   {result['sector']}")
        print(f"RAG used: {result['rag_used']}")
        print(f"Latency:  {result['total_latency_seconds']}s")
        print("\nAnalysis:")
        print(result["analysis"])
        print("=" * 60)