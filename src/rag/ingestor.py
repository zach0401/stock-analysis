from langchain_community.document_loaders import PyPDFLoader, UnstructuredHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import os

CHROMA_DIR = "./data/chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def load_and_chunk_pdf(pdf_path: str) -> list:
    """Load a PDF and split it into chunks."""

    print(f"Loading PDF: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    print(f"  Loaded {len(pages)} pages")

    # RecursiveCharacterTextSplitter tries to split on paragraphs first,
    # then sentences, then words — preserves meaning better than fixed splits
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,        # characters per chunk
        chunk_overlap=50,      # overlap so context isn't lost at boundaries
        separators=["\n\n", "\n", ".", " "]
    )

    chunks = splitter.split_documents(pages)
    print(f"  Split into {len(chunks)} chunks")
    return chunks


def ingest_document(pdf_path: str, collection_name: str = "stock_docs"):
    """Load, chunk, embed, and store a PDF into ChromaDB."""

    chunks = load_and_chunk_pdf(pdf_path)

    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print(f"Embedding and storing {len(chunks)} chunks into ChromaDB...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=CHROMA_DIR
    )

    print(f"Done. Stored in: {CHROMA_DIR}")
    return vectorstore


def load_existing_vectorstore(collection_name: str = "stock_docs"):
    """Load an already-ingested ChromaDB from disk."""

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR
    )
    return vectorstore


if __name__ == "__main__":
    import sys
    docs_folder = "./docs"

    supported = (".pdf", ".htm", ".html")
    files = [f for f in os.listdir(docs_folder) if f.endswith(supported)]

    if not files:
        print("No supported files found in /docs.")
    else:
        for file in files:
            file_path = os.path.join(docs_folder, file)

            # Auto-derive ticker from filename
            # e.g. AAPL_10K.pdf → AAPL
            ticker = file.split("_")[0].upper()
            print(f"\nIngesting {file} as collection: {ticker}")
            ingest_document(file_path, collection_name=ticker)

        print("\nAll documents ingested.")