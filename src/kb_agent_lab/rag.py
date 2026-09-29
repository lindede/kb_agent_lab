"""RAG: chunk docs, persist to Chroma, retrieve with citations metadata."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import chromadb
from chromadb.utils import embedding_functions

from kb_agent_lab import config

_TEXT_SUFFIXES = {".md", ".txt", ".markdown"}


def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    text = text.strip()
    if not text:
        return []
    # Prefer splitting on blank lines / headings for markdown.
    parts = re.split(r"\n(?=#{1,6}\s)|\n{2,}", text)
    parts = [p.strip() for p in parts if p.strip()]
    if not parts:
        parts = [text]

    chunks: list[str] = []
    buf = ""
    for part in parts:
        if not buf:
            buf = part
        elif len(buf) + 1 + len(part) <= chunk_size:
            buf = f"{buf}\n\n{part}"
        else:
            chunks.append(buf)
            # overlap: keep tail of previous buffer
            if overlap > 0 and len(buf) > overlap:
                buf = buf[-overlap:] + "\n\n" + part
            else:
                buf = part
    if buf:
        chunks.append(buf)

    # Hard-split any oversized chunk
    final: list[str] = []
    for c in chunks:
        if len(c) <= chunk_size * 2:
            final.append(c)
            continue
        step = max(chunk_size - overlap, 1)
        for i in range(0, len(c), step):
            final.append(c[i : i + chunk_size])
    return final


def _iter_doc_files(docs_dir: Path) -> list[Path]:
    if not docs_dir.is_dir():
        return []
    files: list[Path] = []
    for p in sorted(docs_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in _TEXT_SUFFIXES and p.name != ".gitkeep":
            files.append(p)
    return files


def _embedding_function():
    # Chat API Key ≠ embedding access. Many gateways return 403 on /embeddings.
    # Default to local ONNX; set EMBEDDING_PROVIDER=openai only if your gateway opens it.
    if config.use_openai_embeddings():
        return embedding_functions.OpenAIEmbeddingFunction(
            api_key=config.OPENAI_API_KEY,
            api_base=config.OPENAI_BASE_URL,
            model_name=config.OPENAI_EMBEDDING_MODEL,
        )
    return embedding_functions.DefaultEmbeddingFunction()


def _embedding_label() -> str:
    if config.use_openai_embeddings():
        return f"openai:{config.OPENAI_EMBEDDING_MODEL}"
    return "local-onnx"


def get_collection(reset: bool = False):
    config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    if reset:
        try:
            client.delete_collection(config.COLLECTION_NAME)
        except Exception:
            pass
    return client.get_or_create_collection(
        name=config.COLLECTION_NAME,
        embedding_function=_embedding_function(),
        metadata={"hnsw:space": "cosine"},
    )


def ingest(docs_dir: Path | None = None, reset: bool = True) -> dict[str, Any]:
    docs_dir = (docs_dir or config.DOCS_DIR).resolve()
    files = _iter_doc_files(docs_dir)
    collection = get_collection(reset=reset)

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict[str, Any]] = []

    for path in files:
        text = path.read_text(encoding="utf-8", errors="ignore")
        rel = str(path.relative_to(docs_dir))
        for i, chunk in enumerate(_chunk_text(text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)):
            digest = hashlib.sha1(f"{rel}:{i}:{chunk[:64]}".encode("utf-8")).hexdigest()[:16]
            ids.append(f"{rel}::{i}::{digest}")
            documents.append(chunk)
            metadatas.append({"source": rel, "chunk": i})

    if documents:
        # Chroma add in batches
        batch = 64
        for start in range(0, len(documents), batch):
            collection.add(
                ids=ids[start : start + batch],
                documents=documents[start : start + batch],
                metadatas=metadatas[start : start + batch],
            )

    return {
        "docs_dir": str(docs_dir),
        "files": len(files),
        "chunks": len(documents),
        "collection": config.COLLECTION_NAME,
        "embedding": _embedding_label(),
    }


def retrieve(query: str, k: int | None = None) -> list[dict[str, Any]]:
    query = (query or "").strip()
    if not query:
        return []
    k = k or config.RETRIEVE_K
    collection = get_collection(reset=False)
    if collection.count() == 0:
        return []

    result = collection.query(query_texts=[query], n_results=min(k, max(collection.count(), 1)))
    docs = (result.get("documents") or [[]])[0]
    metas = (result.get("metadatas") or [[]])[0]
    dists = (result.get("distances") or [[]])[0]
    ids = (result.get("ids") or [[]])[0]

    hits: list[dict[str, Any]] = []
    for i, doc in enumerate(docs):
        meta = metas[i] if i < len(metas) else {}
        hits.append(
            {
                "id": ids[i] if i < len(ids) else "",
                "text": doc,
                "source": meta.get("source", ""),
                "chunk": meta.get("chunk", i),
                "distance": dists[i] if i < len(dists) else None,
            }
        )
    return hits
