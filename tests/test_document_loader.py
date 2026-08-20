import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.document_loader import DocumentLoader


def test_document_loader():
    loader = DocumentLoader(base_data_dir="data")
    documents = loader.load_all_knowledge_sources()

    print("\n" + "=" * 65)
    print("               DOCUMENT LOADER TEST REPORT")
    print("=" * 65)
    print(f"Total Documents Loaded: {len(documents)}\n")

    category_counts = {}
    for idx, doc in enumerate(documents, start=1):
        cat = doc.metadata.get("category", "unknown")
        category_counts[cat] = category_counts.get(cat, 0) + 1

        print(f"[{idx}] Category       : {cat}")
        print(f"    Source Filename : {doc.metadata.get('source')}")
        print(f"    File Type       : {doc.metadata.get('file_type')}")
        print(f"    Content Length  : {len(doc.page_content):,} characters")
        print(f"    Sample Content  : {repr(doc.page_content[:120].strip())}...")
        print("-" * 65)

    print("\nSummary - Documents by Category:")
    for cat, count in category_counts.items():
        print(f"  • {cat}: {count} document(s)")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    test_document_loader()
