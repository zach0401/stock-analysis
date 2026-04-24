from langchain_community.document_loaders import PyPDFLoader, UnstructuredHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import re

CHROMA_DIR = "./data/chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# 10-K section headings to split on first
SECTION_HEADERS = [
    "Item 1.", "Item 1A.", "Item 1B.", "Item 2.", "Item 3.", "Item 4.",
    "Item 5.", "Item 6.", "Item 7.", "Item 7A.", "Item 8.", "Item 9.",
    "Item 9A.", "Item 9B.", "Item 10.", "Item 11.", "Item 12.",
    "Risk Factors", "Business Overview", "Competition", "Properties",
    "Legal Proceedings", "Market for Registrant", "Management",
    "Quantitative and Qualitative", "Financial Statements"
]


def clean_text(text: str) -> str:
    """Remove headers, footers, and junk repeated across pages."""
    # Remove page numbers like "Apple Inc. | 2025 | 42"
    text = re.sub(r'Apple Inc\..*?\d{4}.*?\d+', '', text)
    # Remove excessive whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()


def structure_aware_chunk(pages: list) -> list:
    """
    Split documents by 10-K section headings first,
    then by character limit within each section.
    Every chunk belongs to exactly one section.
    """
    # Step 1 — combine all pages into one text, track page numbers
    full_text = ""
    page_map = {}  # character position → page number

    for page in pages:
        start_pos = len(full_text)
        cleaned = clean_text(page.page_content)
        full_text += cleaned + "\n\n"
        # Map every character in this page to its page number
        for i in range(start_pos, len(full_text)):
            page_map[i] = page.metadata.get("page", 0)

    # Step 2 — split by section headings
    pattern = "|".join([re.escape(h) for h in SECTION_HEADERS])
    sections = re.split(f'({pattern})', full_text)

    # Step 3 — recombine heading with its content
    combined_sections = []
    i = 0
    while i < len(sections):
        if i + 1 < len(sections) and any(sections[i].strip() == h for h in SECTION_HEADERS):
            combined_sections.append(sections[i] + sections[i + 1])
            i += 2
        else:
            if sections[i].strip():
                combined_sections.append(sections[i])
            i += 1

    # Step 4 — chunk within each section
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " "]
    )

    chunks = []
    char_position = 0

    for section in combined_sections:
        # Detect which section heading this belongs to
        section_label = "General"
        for header in SECTION_HEADERS:
            if section.strip().startswith(header):
                section_label = header
                break

        # Split section into chunks
        section_docs = splitter.create_documents([section])

        for doc in section_docs:
            # Find approximate page number from character position
            approx_page = page_map.get(char_position, 0)
            doc.metadata["section"] = section_label
            doc.metadata["page"] = approx_page
            chunks.append(doc)

        char_position += len(section)

    print(f"  Structure-aware chunking: {len(combined_sections)} sections → {len(chunks)} chunks")
    return chunks


def load_file(file_path: str) -> list:
    """Load a PDF or HTML file."""
    print(f"Loading file: {file_path}")

    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".htm") or file_path.endswith(".html"):
        loader = UnstructuredHTMLLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")

    pages = loader.load()
    print(f"  Loaded {len(pages)} pages")
    return pages


def ingest_document(file_path: str, collection_name: str = "stock_docs"):
    """Load, chunk, embed, and store a document into ChromaDB."""

    pages = load_file(file_path)
    chunks = structure_aware_chunk(pages)

    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print(f"Embedding and storing {len(chunks)} chunks into ChromaDB...")

    # Add source metadata to every chunk
    for chunk in chunks:
        chunk.metadata["source"] = file_path
        chunk.metadata["ticker"] = collection_name

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
    docs_folder = "./docs"
    supported = (".pdf", ".htm", ".html")
    files = [f for f in os.listdir(docs_folder) if f.endswith(supported)]

    if not files:
        print("No supported files found in /docs.")
    else:
        for file in files:
            file_path = os.path.join(docs_folder, file)
            ticker = file.split("_")[0].upper()
            print(f"\nIngesting {file} as collection: {ticker}")
            ingest_document(file_path, collection_name=ticker)
        print("\nAll documents ingested.")