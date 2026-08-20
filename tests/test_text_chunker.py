import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.document_loader import DocumentLoader
from app.services.text_chunker import TextChunker


def test_chunker():
    # ── 1. Load all documents ────────────────────────────────────────────────
    loader = DocumentLoader(base_data_dir="data")
    documents = loader.load_all_knowledge_sources()

    # ── 2. Chunk all documents ────────────────────────────────────────────────
    chunker = TextChunker(chunk_size=800, chunk_overlap=100)
    chunks = chunker.chunk_documents(documents)

    # ── 3. Aggregate stats ────────────────────────────────────────────────────
    category_counts: dict[str, int] = {}
    for chunk in chunks:
        cat = chunk.metadata.get("category", "unknown")
        category_counts[cat] = category_counts.get(cat, 0) + 1

    # ── 4. Print report ───────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("               TEXT CHUNKER TEST REPORT")
    print("=" * 65)
    print(f"  Chunk Size    : 800 characters")
    print(f"  Chunk Overlap : 100 characters")
    print(f"  Documents in  : {len(documents)}")
    print(f"  Total Chunks  : {len(chunks)}")

    print("\nChunks by Category:")
    for cat, count in category_counts.items():
        print(f"  • {cat}: {count} chunk(s)")

    # Sample: first chunk of each category
    seen_cats: set[str] = set()
    print("\n" + "-" * 65)
    print("Sample Chunk per Category")
    print("-" * 65)
    for chunk in chunks:
        cat = chunk.metadata.get("category", "unknown")
        if cat in seen_cats:
            continue
        seen_cats.add(cat)

        print(f"\nCategory   : {cat}")
        print(f"Source     : {chunk.metadata.get('source')}")
        print(f"Chunk Index: {chunk.metadata.get('chunk_index')} / {chunk.metadata.get('total_chunks')}")
        print(f"Content Len: {len(chunk.page_content)} chars")
        print(f"Metadata   : {chunk.metadata}")
        print(f"Content    :\n{chunk.page_content[:400]}...")

    print("\n" + "=" * 65 + "\n")


if __name__ == "__main__":
    test_chunker()
