# Stock Analysis Agent (Ollama + RAG)

An interactive stock analysis assistant that combines:

- **Live market data + recent news** via `yfinance`
- **Optional RAG (Retrieval-Augmented Generation)** over ingested 10‑K PDFs using **ChromaDB** + **sentence-transformers** embeddings
- **Local LLM inference** through **Ollama** (OpenAI-compatible endpoint)
- **Two entrypoints**: a CLI (`main.py`) and a Streamlit app (`app.py`)

## What it does

Given a ticker (e.g. `AAPL`) and an optional question, the agent:

1. Fetches **company + financial snapshot** and **recent news**
2. Retrieves **relevant excerpts** from any ingested documents for that ticker (if available)
3. Asks the LLM to generate either:
   - a **targeted answer** (if you ask a question), or
   - a **structured full analysis** (if you leave the question blank)

The output includes **citations** with page numbers when document context is available.

## Requirements

- **Python 3.10+** recommended
- **Ollama installed and running**
  - This project calls an OpenAI-compatible API at `http://localhost:11434/v1`
  - Default model configured in code: `llama3.2` (see `src/llm_engine.py`)

## Setup

Create and activate a virtual environment, then install dependencies.

### Git Bash (Windows)

```bash
python -m venv venv
source venv/Scripts/activate
pip install -r requirements.txt
```

## Run (CLI)

```bash
python main.py
```

You’ll be prompted for:

- ticker symbol (e.g. `AAPL`)
- optional question (press Enter to run full analysis)

## Run (Streamlit UI)

```bash
streamlit run app.py
```

Open the local URL printed in the terminal (typically `http://localhost:8501`).

### Streamlit features

- **Upload a 10‑K PDF** and assign it to a ticker to enable RAG for that ticker
- Run an analysis and view:
  - headline metrics (company / price / sector / RAG active)
  - the generated analysis text
  - a simple **session log** showing tool latencies (observability)

## RAG: ingesting documents

RAG uses ChromaDB persisted under:

- `./data/chroma_db`

You can ingest PDFs in two ways:

- **In the Streamlit sidebar** (upload a PDF and enter its ticker)
- **From the `docs/` folder** using the ingestor script:

```bash
python src/rag/ingestor.py
```

The ingestor will:

- read supported files in `./docs` (PDF/HTML)
- chunk content
- embed with `all-MiniLM-L6-v2`
- store embeddings in Chroma, using the derived ticker as the collection name (e.g. `AAPL_10K.pdf` → `AAPL`)

## Project structure

```text
app.py                  # Streamlit UI entrypoint
main.py                 # CLI entrypoint
src/
  agent.py              # main agent loop (tool orchestration + session log)
  tools.py              # tool wrappers: fetch data, retrieve context, summarize
  data_fetcher.py        # yfinance market/news fetchers
  llm_engine.py          # Ollama/OpenAI client + prompting
  rag/
    ingestor.py          # PDF ingestion (chunk -> embed -> persist in Chroma)
    retriever.py         # MMR retrieval + formatted context with citations
data/
  chroma_db/            # persisted vector DB (generated)
docs/                   # optional input documents (e.g. 10-K PDFs)
```

## Configuration

### LLM / Ollama

In `src/llm_engine.py`:

- **Base URL**: `http://localhost:11434/v1`
- **Model**: `llama3.2`

If you change the model name, make sure it exists locally in Ollama.

### Embeddings / Chroma persistence

In `src/rag/ingestor.py`:

- `CHROMA_DIR = "./data/chroma_db"`
- `EMBEDDING_MODEL = "all-MiniLM-L6-v2"`

## Git hygiene (what not to commit)

Your `.gitignore` is set up to exclude:

- `venv/` (local virtual environment)
- `data/chroma_db/` (generated vector DB files)
- `docs/` (often large PDFs — commit only if you intentionally want them in the repo)
- `.env` (secrets)

## Troubleshooting

### “streamlit: command not found”

- Activate your venv first:

```bash
source venv/Scripts/activate
which streamlit
```

If it’s still missing, reinstall deps:

```bash
pip install -r requirements.txt
```

### “No such command 'app.py'”

Use the correct Streamlit invocation:

```bash
streamlit run app.py
```

### RAG says “No documents have been ingested yet.”

Ingest a PDF (Streamlit sidebar) or run:

```bash
python src/rag/ingestor.py
```

### Hugging Face warning about unauthenticated requests

Embeddings may download from Hugging Face the first time. Optionally set `HF_TOKEN` in your environment to increase rate limits.

## Disclaimer

This project is for educational purposes only and **is not financial advice**. Do your own research.