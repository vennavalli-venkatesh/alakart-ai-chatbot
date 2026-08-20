from typing import Any, Dict, List

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.services.document_loader import Document


class TextChunker:
    """
    Splits general knowledge documents into smaller chunks.

    Knowledge sources:

        products
            -> handled by ProductChunker
            -> one complete chunk per product

        otc
            -> normal text chunking

        wellness
            -> normal text chunking

        navigation
            -> normal text chunking

    All original document metadata is preserved.
    """

    def __init__(
        self,
        chunk_size: int = 700,
        chunk_overlap: int = 100,
    ):

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=[
                "\n\n",
                "\n",
                ". ",
                " ",
                "",
            ],
        )

    # =========================================
    # MAIN CHUNKING METHOD
    # =========================================

    def chunk_documents(
        self,
        documents: List[Document],
    ) -> List[Document]:

        chunks: List[Document] = []

        for doc in documents:

            # =========================================
            # READ SOURCE METADATA
            # =========================================

            category = str(
                doc.metadata.get(
                    "category",
                    "",
                )
            ).strip().lower()

            document_type = str(
                doc.metadata.get(
                    "document_type",
                    "",
                )
            ).strip().lower()

            source_type = str(
                doc.metadata.get(
                    "source_type",
                    "",
                )
            ).strip().lower()

            # =========================================
            # ALAKART PRODUCT
            # =========================================
            #
            # ProductChunker creates one complete
            # Document for each product.
            #
            # Therefore we must NOT split it again.
            # =========================================

            is_product = (
                category == "products"
                or document_type == "product_catalog"
                or source_type == "alakart_catalogue"
            )

            if is_product:

                chunk_metadata: Dict[str, Any] = {
                    **doc.metadata,

                    "chunk_index": 0,
                    "total_chunks": 1,

                    "chunk_type": "complete_product",
                }

                chunks.append(
                    Document(
                        page_content=doc.page_content,
                        metadata=chunk_metadata,
                    )
                )

                continue

            # =========================================
            # GENERAL KNOWLEDGE
            # =========================================
            #
            # OTC
            # Wellness
            # Navigation
            #
            # These documents can be larger, so they
            # are split into smaller retrieval chunks.
            # =========================================

            texts = self._splitter.split_text(
                doc.page_content
            )

            # Remove empty chunks first so that
            # total_chunks is accurate.
            texts = [
                text.strip()
                for text in texts
                if text.strip()
            ]

            total_chunks = len(texts)

            # =========================================
            # CREATE GENERAL KNOWLEDGE CHUNKS
            # =========================================

            for index, text in enumerate(texts):

                chunk_metadata: Dict[str, Any] = {
                    **doc.metadata,

                    "chunk_index": index,
                    "total_chunks": total_chunks,

                    "chunk_type": "general_text",
                }

                chunks.append(
                    Document(
                        page_content=text,
                        metadata=chunk_metadata,
                    )
                )

        # =========================================
        # LOGGING
        # =========================================

        print(
            f"TextChunker: Created "
            f"{len(chunks)} total chunks."
        )

        return chunks