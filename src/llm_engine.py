from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

MODEL = "llama3.2"

SYSTEM_PROMPT = """You are an expert stock analyst. You are given structured financial data about a company,
recent news, and optionally, relevant excerpts from official company documents (10-K filings, annual reports).

Your job is to produce a clear, concise analysis covering:
1. Business overview — describe what the company actually does, its main products and services, and its market position. Use the business_summary field.
2. Key financial signals — reference the exact numbers provided (revenue, margins, PE ratios). Never invent or estimate figures not in the data.
3. Valuation assessment — is the stock cheap, fair, or expensive based on PE vs industry norms?
4. Key risks to watch — use the document excerpts if available, cite page numbers.
5. Overall outlook — bullish / neutral / bearish with a one-line rationale.

Critical rules:
- Only use numbers that appear explicitly in the data provided to you.
- If a document excerpt is relevant, cite it with its page number.
- Never say "trillion" for a company with billion-scale revenue.
- Be specific. Generic analysis is not acceptable.

Always end with: DISCLAIMER: This is not financial advice. Do your own research."""


def analyze_stock(stock_data: dict, news: list[dict], rag_context: str = None, question: str = None) -> str:

    news_text = "\n".join([f"- {n['title']} ({n['publisher']})" for n in news])

    rag_section = ""
    if rag_context and "No relevant" not in rag_context and "Could not" not in rag_context:
        rag_section = f"""
RELEVANT EXCERPTS FROM OFFICIAL DOCUMENTS:
{rag_context}
"""

    # Switch between targeted question and full analysis
    if question:
        task = f"""The user has a specific question: "{question}"

Answer it directly and specifically using the financial data and document excerpts provided.
Keep your response focused — don't produce a full analysis, just answer the question with supporting evidence and cite page numbers where relevant."""
    else:
        task = """Produce a full structured analysis covering:
1. Business overview — actual products, services, market position
2. Key financial signals — use the exact numbers provided
3. Valuation assessment — PE ratio vs industry norms
4. Key risks to watch — cite document page numbers if available
5. Overall outlook — bullish / neutral / bearish with one-line rationale"""

    user_message = f"""
FINANCIAL DATA:
{stock_data}

RECENT NEWS:
{news_text}
{rag_section}

TASK:
{task}
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ],
        temperature=0.3,
        max_tokens=800,
    )

    return response.choices[0].message.content