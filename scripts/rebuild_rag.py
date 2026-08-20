import os
import sys
import uuid

# =========================================
# PROJECT ROOT
# =========================================

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(
        encoding="utf-8"
    )


# =========================================
# SERVICES
# =========================================

from app.services.document_loader import DocumentLoader
from app.services.data_cleaner import DataCleaner
from app.services.text_chunker import TextChunker
from app.services.product_chunker import ProductChunker
from app.services.embedding_service import EmbeddingService
from app.services.vector_store import VectorStore


# =========================================
# KNOWLEDGE CATEGORIES
# =========================================

KNOWLEDGE_CATEGORIES = [
    "navigation",
    "otc",
    "products",
    "wellness",
]


# =========================================
# REBUILD INDEX
# =========================================

def rebuild_index():

    print("=" * 70)
    print("STARTING ALAKART RAG PIPELINE REBUILD")
    print("=" * 70)

    # =========================================
    # STEP 1
    # LOAD ALL KNOWLEDGE SOURCES
    # =========================================

    print("\n" + "=" * 70)
    print("STEP 1: LOADING KNOWLEDGE SOURCES")
    print("=" * 70)

    loader = DocumentLoader(
        base_data_dir="data"
    )

    documents = loader.load_all_knowledge_sources(
        categories=KNOWLEDGE_CATEGORIES
    )

    print(
        f"\nTotal documents/pages loaded: "
        f"{len(documents)}"
    )

    if not documents:

        print(
            "\nERROR: No documents were loaded."
        )

        print(
            "Please check your data/ folder."
        )

        return

    # =========================================
    # CATEGORY STATISTICS
    # =========================================

    category_counts = {}

    source_type_counts = {}

    document_type_counts = {}

    ocr_pages = 0

    for doc in documents:

        category = str(
            doc.metadata.get(
                "category",
                "unknown"
            )
        ).strip().lower()

        source_type = str(
            doc.metadata.get(
                "source_type",
                "unknown"
            )
        ).strip().lower()

        document_type = str(
            doc.metadata.get(
                "document_type",
                "unknown"
            )
        ).strip().lower()

        category_counts[category] = (
            category_counts.get(
                category,
                0
            ) + 1
        )

        source_type_counts[source_type] = (
            source_type_counts.get(
                source_type,
                0
            ) + 1
        )

        document_type_counts[document_type] = (
            document_type_counts.get(
                document_type,
                0
            ) + 1
        )

        if doc.metadata.get(
            "ocr_used",
            False
        ):
            ocr_pages += 1

    print("\nCategory breakdown:")

    for category, count in sorted(
        category_counts.items()
    ):

        print(
            f"  {category:<15} {count}"
        )

    print("\nSource type breakdown:")

    for source_type, count in sorted(
        source_type_counts.items()
    ):

        print(
            f"  {source_type:<25} {count}"
        )

    print("\nDocument type breakdown:")

    for document_type, count in sorted(
        document_type_counts.items()
    ):

        print(
            f"  {document_type:<25} {count}"
        )

    print(
        f"\nOCR pages: {ocr_pages}"
    )

    # =========================================
    # STEP 2
    # CLEAN DOCUMENTS
    # =========================================

    print("\n" + "=" * 70)
    print("STEP 2: CLEANING DOCUMENTS")
    print("=" * 70)

    cleaner = DataCleaner()

    cleaned_documents = (
        cleaner.clean_documents(
            documents
        )
    )

    print(
        f"Documents after cleaning: "
        f"{len(cleaned_documents)}"
    )

    if not cleaned_documents:

        print(
            "\nERROR: Cleaning produced "
            "zero documents."
        )

        return

    # =========================================
    # STEP 3
    # CREATE CHUNKS
    # =========================================

    print("\n" + "=" * 70)
    print("STEP 3: CREATING RAG CHUNKS")
    print("=" * 70)

    product_chunker = ProductChunker()

    text_chunker = TextChunker(
        chunk_size=700,
        chunk_overlap=100,
    )

    all_chunks = []

    products_detected = []

    chunk_category_counts = {}

    chunk_type_counts = {}

    # =========================================
    # PROCESS EACH DOCUMENT
    # =========================================

    for doc in cleaned_documents:

        category = str(
            doc.metadata.get(
                "category",
                ""
            )
        ).strip().lower()

        # =========================================
        # ALAKART PRODUCTS
        # =========================================

        if product_chunker.is_product_document(
            doc
        ):

            product_chunks = (
                product_chunker
                .chunk_product_document(doc)
            )

            all_chunks.extend(
                product_chunks
            )

            # -----------------------------------------
            # Track products
            # -----------------------------------------

            for chunk in product_chunks:

                product_name = str(
                    chunk.metadata.get(
                        "product_name",
                        ""
                    )
                ).strip()

                if (
                    product_name
                    and product_name
                    not in products_detected
                ):

                    products_detected.append(
                        product_name
                    )

        # =========================================
        # GENERAL KNOWLEDGE
        # =========================================

        else:

            general_chunks = (
                text_chunker
                .chunk_documents([doc])
            )

            all_chunks.extend(
                general_chunks
            )

    # =========================================
    # CHUNK STATISTICS
    # =========================================

    for chunk in all_chunks:

        category = str(
            chunk.metadata.get(
                "category",
                "unknown"
            )
        ).strip().lower()

        chunk_type = str(
            chunk.metadata.get(
                "chunk_type",
                "unknown"
            )
        ).strip().lower()

        chunk_category_counts[
            category
        ] = (
            chunk_category_counts.get(
                category,
                0
            ) + 1
        )

        chunk_type_counts[
            chunk_type
        ] = (
            chunk_type_counts.get(
                chunk_type,
                0
            ) + 1
        )

    print(
        f"\nTotal chunks created: "
        f"{len(all_chunks)}"
    )

    print("\nChunks by knowledge category:")

    for category, count in sorted(
        chunk_category_counts.items()
    ):

        print(
            f"  {category:<15} {count}"
        )

    print("\nChunks by chunk type:")

    for chunk_type, count in sorted(
        chunk_type_counts.items()
    ):

        print(
            f"  {chunk_type:<25} {count}"
        )

    print(
        f"\nAlakart products detected: "
        f"{len(products_detected)}"
    )

    for product_name in products_detected:

        print(
            f"  - {product_name}"
        )

    if not all_chunks:

        print(
            "\nERROR: No chunks were created."
        )

        return

    # =========================================
    # STEP 4
    # GENERATE EMBEDDINGS
    # =========================================

    print("\n" + "=" * 70)
    print("STEP 4: GENERATING EMBEDDINGS")
    print("=" * 70)

    print(
        "\nLoading embedding model..."
    )

    embedder = EmbeddingService()

    print(
        f"Embedding model: "
        f"{embedder.model_name}"
    )

    print(
        f"Embedding dimension: "
        f"{embedder.embedding_dimension}"
    )

    print(
        "\nGenerating embeddings..."
    )

    embedded_results = (
        embedder.embed_documents(
            all_chunks
        )
    )

    print(
        f"Embeddings generated: "
        f"{len(embedded_results)}"
    )

    if not embedded_results:

        print(
            "\nERROR: No embeddings generated."
        )

        return

    # =========================================
    # STEP 5
    # CREATE VECTOR STORE
    # =========================================

    print("\n" + "=" * 70)
    print("STEP 5: REBUILDING CHROMA VECTOR STORE")
    print("=" * 70)

    vector_store = VectorStore(
        persist_directory="chroma_db",
        collection_name="rag_collection",
    )

    # =========================================
    # DELETE OLD INDEX
    # =========================================

    print(
        "\nDeleting old Chroma collection..."
    )

    vector_store.reset_collection()

    print(
        "Old collection removed."
    )

    # =========================================
    # PREPARE VECTORS
    # =========================================

    ids = []
    embeddings = []
    texts = []
    metadatas = []

    for index, result in enumerate(
        embedded_results
    ):

        chunk_id = (
            f"chunk_"
            f"{uuid.uuid4().hex[:12]}_"
            f"{index}"
        )

        ids.append(
            chunk_id
        )

        embeddings.append(
            result["embedding"]
        )

        texts.append(
            result["page_content"]
        )

        metadata = (
            result["metadata"].copy()
        )

        # -----------------------------------------
        # Store unique chunk ID
        # -----------------------------------------

        metadata["chunk_id"] = (
            chunk_id
        )

        metadatas.append(
            metadata
        )

    # =========================================
    # BATCH INSERT
    # =========================================

    batch_size = 1000

    total_batches = (
        (
            len(ids)
            + batch_size
            - 1
        )
        // batch_size
    )

    print(
        f"\nTotal vectors: {len(ids)}"
    )

    print(
        f"Batch size: {batch_size}"
    )

    print(
        f"Total batches: {total_batches}"
    )

    for batch_number, start in enumerate(
        range(
            0,
            len(ids),
            batch_size
        ),
        start=1,
    ):

        end = min(
            start + batch_size,
            len(ids)
        )

        print(
            f"\nAdding batch "
            f"{batch_number}/{total_batches} "
            f"({start} - {end - 1})"
        )

        vector_store.add_documents(
            ids=ids[start:end],
            embeddings=embeddings[start:end],
            documents=texts[start:end],
            metadatas=metadatas[start:end],
        )

    # =========================================
    # FINAL COUNT
    # =========================================

    final_count = (
        vector_store
        .get_collection_count()
    )

    # =========================================
    # FINAL VALIDATION
    # =========================================

    print("\n" + "=" * 70)
    print("RAG REBUILD COMPLETE")
    print("=" * 70)

    print(
        f"Documents loaded:       "
        f"{len(documents)}"
    )

    print(
        f"Documents cleaned:      "
        f"{len(cleaned_documents)}"
    )

    print(
        f"Chunks created:         "
        f"{len(all_chunks)}"
    )

    print(
        f"Embeddings generated:   "
        f"{len(embedded_results)}"
    )

    print(
        f"Vectors stored:         "
        f"{final_count}"
    )

    print(
        f"Alakart products:       "
        f"{len(products_detected)}"
    )

    print(
        f"Embedding model:        "
        f"{embedder.model_name}"
    )

    print(
        f"Embedding dimension:    "
        f"{embedder.embedding_dimension}"
    )

    print(
        f"Collection:             "
        f"rag_collection"
    )

    print(
        f"Vector store status:    "
        f"{'SUCCESS' if final_count > 0 else 'EMPTY'}"
    )

    # =========================================
    # CATEGORY SUMMARY
    # =========================================

    print("\nKnowledge categories:")

    for category, count in sorted(
        chunk_category_counts.items()
    ):

        print(
            f"  {category:<15} {count}"
        )

    # =========================================
    # PRODUCT SUMMARY
    # =========================================

    print("\nAlakart products:")

    if products_detected:

        for product_name in products_detected:

            print(
                f"  - {product_name}"
            )

    else:

        print(
            "  No products detected."
        )

    print("=" * 70)

    # =========================================
    # IMPORTANT
    # =========================================
    #
    # We intentionally do NOT automatically run
    # test_retrieval.py here.
    #
    # Retrieval validation will be performed
    # separately after we update the Retriever.
    #
    # This prevents an old test script from hiding
    # problems in the new pipeline.
    # =========================================

    print(
        "\nIndex rebuild finished successfully."
    )

    print(
        "Next step: update and validate Retriever."
    )


# =========================================
# MAIN
# =========================================

if __name__ == "__main__":
    rebuild_index()