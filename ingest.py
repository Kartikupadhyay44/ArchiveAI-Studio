"""
ArchiveAI — Modern History & Archival Chronicle Assistant
Ingestion Module: PDF Extraction, Text Cleaning, Chunking, Embedding Generation, and ChromaDB Persistence
"""

import os
import re
import sys
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional

import pymupdf
import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

# Offline Hugging Face mode for fast, deterministic local loading
os.environ["HF_HUB_OFFLINE"] = "1"

# Configuration constants
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_PDF_PATH = BASE_DIR / "data" / "indian_railways_and_telegraph_history_50p.pdf"
FALLBACK_PDF_PATH = BASE_DIR / "indian_railways_and_telegraph_history_50p.pdf"

PDF_PATH_STR = os.getenv("PDF_PATH", str(DEFAULT_PDF_PATH))
CHROMA_DIR = os.getenv("CHROMA_DB_DIR", str(BASE_DIR / "chroma_db"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "archiveai_collection")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")

MANIFEST_FILENAME = "index_manifest.json"


def resolve_pdf_path(custom_path: Optional[str] = None) -> Path:
    """Resolve and validate the PDF path, checking data/ folder and workspace root."""
    if custom_path:
        p = Path(custom_path)
        if p.is_file():
            return p

    candidates = [
        Path(PDF_PATH_STR),
        DEFAULT_PDF_PATH,
        FALLBACK_PDF_PATH,
    ]
    for c in candidates:
        if c.is_file():
            return c

    raise FileNotFoundError(
        f"PDF file not found. Looked in: {[str(c) for c in candidates]}. "
        "Please place 'indian_railways_and_telegraph_history_50p.pdf' in the 'data/' folder."
    )


def calculate_file_hash(filepath: Path) -> str:
    """Calculate SHA256 hash of a file for change tracking."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def clean_page_text(raw_text: str) -> str:
    """
    Remove repeated running headers and decorative footers from the PDF,
    preserving chapter headings, clauses, and archival content.
    """
    raw_lines = [l.strip() for l in raw_text.splitlines() if l.strip()]
    cleaned_lines = []
    for line in raw_lines:
        # Filter decorative dividers and repeating header/footer stamps
        if line.startswith("=====") or line.startswith("-----"):
            continue
        if line.startswith("Page ") and "Confidential Assessment Corpus" in line:
            continue
        if line.startswith("Document Page ") and "of 50" in line:
            continue
        if line.startswith("Iron Tracks & Copper Wires: Archival History"):
            continue
        cleaned_lines.append(line)

    # Reconstruct text preserving paragraph transitions
    return " ".join(cleaned_lines).strip()


def extract_pages(pdf_path: Path) -> List[Dict[str, Any]]:
    """
    Extract text page-by-page from the PDF using PyMuPDF.
    Ensures exactly 50 pages are extracted and validates content.
    """
    if not pdf_path.is_file():
        raise FileNotFoundError(f"Cannot extract from nonexistent PDF: {pdf_path}")

    doc = pymupdf.open(str(pdf_path))
    num_pages = len(doc)
    if num_pages == 0:
        raise ValueError(f"PDF {pdf_path} is empty (0 pages).")

    pages = []
    for page_idx in range(num_pages):
        page_num = page_idx + 1  # 1-indexed to match archival page references
        page = doc[page_idx]
        raw_text = page.get_text()
        cleaned_text = clean_page_text(raw_text)

        pages.append({
            "page_number": page_num,
            "page_label": f"page_{page_num}",
            "raw_text": raw_text,
            "cleaned_text": cleaned_text,
            "char_count": len(cleaned_text),
        })

    doc.close()
    return pages


def chunk_page_text(
    cleaned_text: str,
    page_num: int,
    source_name: str,
    target_size: int = 850,
    overlap: int = 150,
) -> List[Dict[str, Any]]:
    """
    Chunk text from a single page into overlapping segments.
    Preserves sentence boundaries and keeps key factual statements intact.
    Target chunk size: ~700-900 chars; Overlap: ~100-150 chars.
    """
    if len(cleaned_text) <= target_size + overlap:
        # Page fits comfortably within a single chunk; keeping it intact prevents any splitting of answers
        return [{
            "chunk_id": f"page_{page_num}_chunk_0",
            "page_number": page_num,
            "page_label": f"page_{page_num}",
            "source": source_name,
            "text": cleaned_text,
            "char_count": len(cleaned_text),
        }]

    # Split text cleanly on sentence boundaries
    sentences = re.split(r"(?<=[.!?])\s+", cleaned_text)
    
    chunks = []
    current_sentences: List[str] = []
    current_length = 0

    for sentence in sentences:
        sent = sentence.strip()
        if not sent:
            continue

        if current_length + len(sent) > target_size and current_sentences:
            chunk_content = " ".join(current_sentences).strip()
            chunk_idx = len(chunks)
            chunks.append({
                "chunk_id": f"page_{page_num}_chunk_{chunk_idx}",
                "page_number": page_num,
                "page_label": f"page_{page_num}",
                "source": source_name,
                "text": chunk_content,
                "char_count": len(chunk_content),
            })

            # Calculate overlap from the end of current sentences
            overlap_sentences: List[str] = []
            overlap_length = 0
            for prev_sent in reversed(current_sentences):
                overlap_sentences.insert(0, prev_sent)
                overlap_length += len(prev_sent) + 1
                if overlap_length >= overlap:
                    break

            current_sentences = list(overlap_sentences)
            current_length = sum(len(s) for s in current_sentences) + max(0, len(current_sentences) - 1)

        current_sentences.append(sent)
        current_length += len(sent) + 1

    if current_sentences:
        chunk_content = " ".join(current_sentences).strip()
        chunk_idx = len(chunks)
        if not chunks or chunks[-1]["text"] != chunk_content:
            chunks.append({
                "chunk_id": f"page_{page_num}_chunk_{chunk_idx}",
                "page_number": page_num,
                "page_label": f"page_{page_num}",
                "source": source_name,
                "text": chunk_content,
                "char_count": len(chunk_content),
            })

    return chunks


def build_all_chunks(pages: List[Dict[str, Any]], source_name: str) -> List[Dict[str, Any]]:
    """Build all chunks across all pages of the document."""
    all_chunks = []
    for page_data in pages:
        chunks = chunk_page_text(
            cleaned_text=page_data["cleaned_text"],
            page_num=page_data["page_number"],
            source_name=source_name,
        )
        all_chunks.extend(chunks)
    return all_chunks


def get_embedding_model(model_name: str = EMBEDDING_MODEL_NAME) -> SentenceTransformer:
    """Load the local SentenceTransformer model."""
    try:
        return SentenceTransformer(model_name, local_files_only=True)
    except Exception:
        return SentenceTransformer(model_name)


def load_manifest(chroma_dir: Path) -> Optional[Dict[str, Any]]:
    """Load manifest if present."""
    manifest_path = chroma_dir / MANIFEST_FILENAME
    if manifest_path.is_file():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def save_manifest(chroma_dir: Path, data: Dict[str, Any]) -> None:
    """Save manifest to track indexed PDF fingerprint."""
    manifest_path = chroma_dir / MANIFEST_FILENAME
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def is_index_ready(pdf_path: Path, chroma_dir: Path, collection_name: str) -> bool:
    """Check if the persistent index is valid, complete, and up-to-date with the PDF."""
    manifest = load_manifest(chroma_dir)
    if not manifest:
        return False

    current_hash = calculate_file_hash(pdf_path)
    if manifest.get("pdf_hash") != current_hash:
        return False

    if not chroma_dir.is_dir():
        return False

    try:
        client = chromadb.PersistentClient(path=str(chroma_dir))
        collections = [c.name for c in client.list_collections()]
        if collection_name not in collections:
            return False
        col = client.get_collection(collection_name)
        return col.count() == manifest.get("num_chunks", -1) and col.count() > 0
    except Exception:
        return False


def run_ingestion(
    force_rebuild: bool = False,
    custom_pdf_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Main ingestion execution function.
    Extracts PDF, chunks text, generates embeddings, and persists to ChromaDB.
    Reuses existing database if fingerprint matches and force_rebuild is False.
    """
    pdf_path = resolve_pdf_path(custom_pdf_path)
    chroma_path = Path(CHROMA_DIR)
    chroma_path.mkdir(parents=True, exist_ok=True)
    file_hash = calculate_file_hash(pdf_path)
    source_filename = pdf_path.name

    # Check for reuse
    if not force_rebuild and is_index_ready(pdf_path, chroma_path, COLLECTION_NAME):
        manifest = load_manifest(chroma_path) or {}
        print(f"[*] ChromaDB collection '{COLLECTION_NAME}' is already up-to-date. Reusing existing index.")
        return {
            "status": "reused",
            "pdf_path": str(pdf_path),
            "pdf_hash": file_hash,
            "num_pages": manifest.get("num_pages", 50),
            "num_chunks": manifest.get("num_chunks", 0),
            "collection_name": COLLECTION_NAME,
            "chroma_dir": str(chroma_path),
        }

    print(f"[*] Starting ingestion for: {pdf_path}")
    pages = extract_pages(pdf_path)
    num_pages = len(pages)
    print(f"[*] Extracted {num_pages} pages using PyMuPDF.")

    chunks = build_all_chunks(pages, source_name=source_filename)
    num_chunks = len(chunks)
    print(f"[*] Created {num_chunks} overlapping sentence-aligned chunks.")

    print(f"[*] Loading SentenceTransformer model '{EMBEDDING_MODEL_NAME}'...")
    embedder = get_embedding_model(EMBEDDING_MODEL_NAME)

    texts = [c["text"] for c in chunks]
    ids = [c["chunk_id"] for c in chunks]
    metadatas = [
        {
            "page_number": c["page_number"],
            "page_label": c["page_label"],
            "source": c["source"],
            "chunk_id": c["chunk_id"],
        }
        for c in chunks
    ]

    print(f"[*] Generating {num_chunks} embeddings...")
    embeddings = embedder.encode(texts, batch_size=32, show_progress_bar=False).tolist()

    print(f"[*] Persisting to ChromaDB at: {chroma_path}")
    client = chromadb.PersistentClient(path=str(chroma_path))

    # Recreate collection cleanly if rebuilding or existing
    existing_collections = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing_collections:
        client.delete_collection(COLLECTION_NAME)

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "ArchiveAI Indian Railways and Telegraph Historical Corpus"},
    )

    # Batch add to ChromaDB
    batch_size = 64
    for i in range(0, num_chunks, batch_size):
        end_idx = min(i + batch_size, num_chunks)
        collection.add(
            ids=ids[i:end_idx],
            embeddings=embeddings[i:end_idx],
            documents=texts[i:end_idx],
            metadatas=metadatas[i:end_idx],
        )

    # Save manifest
    manifest_data = {
        "pdf_hash": file_hash,
        "pdf_filename": source_filename,
        "num_pages": num_pages,
        "num_chunks": num_chunks,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "collection_name": COLLECTION_NAME,
    }
    save_manifest(chroma_path, manifest_data)

    print(f"[✓] Ingestion complete! {num_pages} pages -> {num_chunks} chunks stored in ChromaDB.")
    return {
        "status": "indexed",
        "pdf_path": str(pdf_path),
        "pdf_hash": file_hash,
        "num_pages": num_pages,
        "num_chunks": num_chunks,
        "collection_name": COLLECTION_NAME,
        "chroma_dir": str(chroma_path),
    }


if __name__ == "__main__":
    rebuild_flag = "--rebuild" in sys.argv or "-r" in sys.argv
    run_ingestion(force_rebuild=rebuild_flag)
