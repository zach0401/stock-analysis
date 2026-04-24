import json
import time
import os
import sys

# Make sure src is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.rag.retriever import get_relevant_context

DATASET_PATH = "./src/eval/eval_dataset.json"


def score_retrieval(context: str, question_data: dict) -> dict:
    """
    Score a single retrieval result against expected values.
    Returns scores for keyword presence and page range accuracy.
    """
    context_lower = context.lower()

    # Keyword score — how many expected keywords appear in retrieved chunks
    expected_keywords = question_data.get("expected_keywords", [])
    found_keywords = [kw for kw in expected_keywords if kw.lower() in context_lower]
    keyword_score = len(found_keywords) / len(expected_keywords) if expected_keywords else 0

    # Page score — did we retrieve from the expected page range
    min_page = question_data.get("min_expected_page", 0)
    max_page = question_data.get("max_expected_page", 999)

    # Extract page numbers from context
    import re
    pages_found = [int(p) for p in re.findall(r'Page (\d+)', context)]
    page_in_range = any(min_page <= p <= max_page for p in pages_found)
    page_score = 1.0 if page_in_range else 0.0

    # Section score — did we retrieve from the expected section
    expected_section = question_data.get("expected_section", "")
    section_score = 1.0 if expected_section.lower() in context_lower else 0.0

    # Overall score — weighted average
    overall = (keyword_score * 0.5) + (page_score * 0.3) + (section_score * 0.2)

    return {
        "keyword_score": round(keyword_score, 2),
        "page_score": page_score,
        "section_score": section_score,
        "overall_score": round(overall, 2),
        "found_keywords": found_keywords,
        "missing_keywords": [kw for kw in expected_keywords if kw.lower() not in context_lower],
        "pages_retrieved": pages_found,
    }


def run_eval():
    """Run all questions through the RAG pipeline and score results."""

    with open(DATASET_PATH, "r") as f:
        dataset = json.load(f)

    ticker = dataset["ticker"]
    questions = dataset["questions"]

    print(f"\nRunning eval for {ticker} — {len(questions)} questions")
    print("=" * 70)

    results = []
    total_latency = 0

    for q in questions:
        print(f"\n[{q['id']}] {q['question']}")

        start = time.time()
        context = get_relevant_context(q["question"], ticker=ticker, k=4)
        latency = round(time.time() - start, 2)
        total_latency += latency

        scores = score_retrieval(context, q)
        scores["question_id"] = q["id"]
        scores["question"] = q["question"]
        scores["latency_seconds"] = latency

        results.append(scores)

        # Print per-question result
        status = "PASS" if scores["overall_score"] >= 0.5 else "FAIL"
        print(f"  Status:   {status}")
        print(f"  Overall:  {scores['overall_score']} | Keywords: {scores['keyword_score']} | Page: {scores['page_score']} | Section: {scores['section_score']}")
        print(f"  Found:    {scores['found_keywords']}")
        print(f"  Missing:  {scores['missing_keywords']}")
        print(f"  Pages:    {scores['pages_retrieved']}")
        print(f"  Latency:  {latency}s")

    # Summary
    print("\n" + "=" * 70)
    print("EVAL SUMMARY")
    print("=" * 70)

    passing = [r for r in results if r["overall_score"] >= 0.5]
    avg_score = round(sum(r["overall_score"] for r in results) / len(results), 2)
    avg_keyword = round(sum(r["keyword_score"] for r in results) / len(results), 2)
    avg_latency = round(total_latency / len(results), 2)

    print(f"Questions:      {len(questions)}")
    print(f"Passing (>=0.5): {len(passing)}/{len(questions)}")
    print(f"Avg score:      {avg_score}")
    print(f"Avg keyword:    {avg_keyword}")
    print(f"Avg latency:    {avg_latency}s")

    # Save results to file
    output_path = "./src/eval/eval_results.json"
    with open(output_path, "w") as f:
        json.dump({
            "summary": {
                "total_questions": len(questions),
                "passing": len(passing),
                "avg_score": avg_score,
                "avg_keyword_score": avg_keyword,
                "avg_latency_seconds": avg_latency
            },
            "results": results
        }, f, indent=2)

    print(f"\nFull results saved to: {output_path}")
    return results


if __name__ == "__main__":
    run_eval()