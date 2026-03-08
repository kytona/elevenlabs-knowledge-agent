from __future__ import annotations

import argparse
from pathlib import Path

from app.config import get_openai_client, get_qdrant_client, get_settings
from app.rag import chunk_text, ingest_chunks


SUPPORTED_SUFFIXES = {".md", ".txt"}


def load_text_files(path: Path) -> list[tuple[str, str]]:
    if path.is_file():
        candidates = [path]
    else:
        candidates = sorted(file for file in path.rglob("*") if file.suffix.lower() in SUPPORTED_SUFFIXES)

    documents: list[tuple[str, str]] = []
    for file_path in candidates:
        if file_path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        documents.append((str(file_path), file_path.read_text(encoding="utf-8")))
    return documents


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest markdown or text documents into Qdrant.")
    parser.add_argument("path", help="Path to a markdown/text file or a directory of files.")
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--chunk-overlap", type=int, default=120)
    parser.add_argument("--recreate-collection", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    openai_client = get_openai_client()
    qdrant_client = get_qdrant_client()

    source_path = Path(args.path)
    if not source_path.exists():
        raise SystemExit(f"Path not found: {source_path}")

    documents = load_text_files(source_path)
    if not documents:
        raise SystemExit("No supported .md or .txt documents found.")

    total_points = 0
    recreate = args.recreate_collection
    for document_path, text in documents:
        chunks = chunk_text(text, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
        inserted = ingest_chunks(
            client=qdrant_client,
            openai_client=openai_client,
            settings=settings,
            source=document_path,
            chunks=chunks,
            recreate_collection=recreate,
        )
        recreate = False
        total_points += inserted
        print(f"Ingested {inserted} chunks from {document_path}")

    print(f"Done. Upserted {total_points} chunks into '{settings.qdrant_collection_name}'.")
    qdrant_client.close()


if __name__ == "__main__":
    main()

