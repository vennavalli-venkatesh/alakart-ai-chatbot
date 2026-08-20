from typing import List, Dict, Any, Optional

from app.services.embedding_service import EmbeddingService
from app.services.vector_store import VectorStore


class Retriever:
    """
    Retrieves relevant knowledge for SITA.

    Retrieval strategy:

    1. General health / OTC information
    2. Alakart product information

    For health-related questions, both types are retrieved so that
    RAGService can build the required response:

        General medicine guidance
                ↓
        Alakart product suggestion
                ↓
        Safety / Note

    This class does NOT generate answers.
    It only retrieves and classifies knowledge.
    """

    def __init__(self, top_k: int = 5):

        self.top_k = top_k

        self.embedding_service = EmbeddingService()

        self.vector_store = VectorStore()

    # =========================================================
    # MAIN RETRIEVAL
    # =========================================================

    def retrieve(
        self,
        query: str,
        intent: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieves both general health information and Alakart
        product information when appropriate.

        The returned list contains structured chunks with:

        - content
        - metadata
        - distance
        """

        query = query.strip()

        if not query:
            return []

        print("\n" + "=" * 60)
        print("RETRIEVER")
        print("=" * 60)
        print(f"Query: {query}")

        # -----------------------------------------------------
        # Determine retrieval requirements
        # -----------------------------------------------------

        retrieval_plan = self._build_retrieval_plan(
            query=query,
            intent=intent
        )

        print(
            f"Retrieve general information: "
            f"{retrieval_plan['general']}"
        )

        print(
            f"Retrieve Alakart products: "
            f"{retrieval_plan['alakart']}"
        )

        # -----------------------------------------------------
        # Query embedding
        # -----------------------------------------------------

        query_embedding = (
            self.embedding_service.embed_text(query)
        )

        # -----------------------------------------------------
        # General knowledge retrieval
        # -----------------------------------------------------

        general_chunks = []

        if retrieval_plan["general"]:

            general_chunks = self._search_candidates(
                query_embedding=query_embedding,
                candidate_k=20
            )

            general_chunks = [
                chunk
                for chunk in general_chunks
                if not self._is_alakart_product(
                    chunk.get("metadata", {})
                )
            ]

        print(
            f"General chunks found: "
            f"{len(general_chunks)}"
        )

        # -----------------------------------------------------
        # Alakart retrieval
        #
        # Use a product-focused query embedding.
        #
        # This is important.
        #
        # Example:
        #
        # User:
        # "I have fever and cough"
        #
        # Product query:
        # "Alakart product relevant to fever and cough"
        #
        # This gives the vector search a stronger signal that
        # we are looking for an Alakart product.
        # -----------------------------------------------------

        alakart_chunks = []

        if retrieval_plan["alakart"]:

            product_query = (
                "Alakart product relevant to: "
                + query
            )

            product_embedding = (
                self.embedding_service.embed_text(
                    product_query
                )
            )

            product_candidates = self._search_candidates(
                query_embedding=product_embedding,
                candidate_k=20
            )

            alakart_chunks = [
                chunk
                for chunk in product_candidates
                if self._is_alakart_product(
                    chunk.get("metadata", {})
                )
            ]

        print(
            f"Alakart chunks found: "
            f"{len(alakart_chunks)}"
        )

        # -----------------------------------------------------
        # Remove duplicate chunks
        # -----------------------------------------------------

        combined = self._merge_unique_chunks(
            general_chunks,
            alakart_chunks
        )

        # -----------------------------------------------------
        # Keep useful number of chunks
        # -----------------------------------------------------

        general_final = [
            chunk
            for chunk in combined
            if not self._is_alakart_product(
                chunk.get("metadata", {})
            )
        ][:self.top_k]

        alakart_final = [
            chunk
            for chunk in combined
            if self._is_alakart_product(
                chunk.get("metadata", {})
            )
        ][:self.top_k]

        # -----------------------------------------------------
        # IMPORTANT ORDER
        #
        # General information first.
        # Alakart products second.
        #
        # RAGService will use this structure to create the
        # final answer in the correct order.
        # -----------------------------------------------------

        final_chunks = (
            general_final +
            alakart_final
        )

        print(
            f"Final general chunks: "
            f"{len(general_final)}"
        )

        print(
            f"Final Alakart chunks: "
            f"{len(alakart_final)}"
        )

        print("=" * 60 + "\n")

        return final_chunks

    # =========================================================
    # RETRIEVAL PLAN
    # =========================================================

    def _build_retrieval_plan(
        self,
        query: str,
        intent: Optional[Dict[str, Any]]
    ) -> Dict[str, bool]:
        """
        Determines which knowledge categories should be retrieved.

        RAGService can later provide a stronger intent classification.

        Until then, this method safely defaults health questions
        to retrieving BOTH general information and Alakart products.
        """

        # -----------------------------------------------------
        # If RAGService already detected intent, respect it.
        # -----------------------------------------------------

        if intent:

            asks_alakart = bool(
                intent.get(
                    "asks_alakart",
                    False
                )
            )

            asks_general = bool(
                intent.get(
                    "asks_general_medicine",
                    False
                )
            )

            # User explicitly asks about Alakart.
            if asks_alakart:

                return {
                    "general": True,
                    "alakart": True,
                }

            # User asks general health / medicine.
            #
            # We still retrieve Alakart because the required
            # SITA workflow should be able to provide an Alakart
            # option after general guidance.
            if asks_general:

                return {
                    "general": True,
                    "alakart": True,
                }

        # -----------------------------------------------------
        # Default behaviour
        #
        # Retrieve BOTH.
        #
        # This prevents the previous problem where a general
        # health query only retrieved general medicine chunks.
        # -----------------------------------------------------

        return {
            "general": True,
            "alakart": True,
        }

    # =========================================================
    # VECTOR SEARCH
    # =========================================================

    def _search_candidates(
        self,
        query_embedding: List[float],
        candidate_k: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Performs vector search and converts the ChromaDB
        response into our standard chunk structure.
        """

        results = self.vector_store.search(
            query_embedding,
            top_k=candidate_k
        )

        if (
            not results
            or not results.get("documents")
            or not results["documents"][0]
        ):
            return []

        documents = results["documents"][0]

        metadatas = (
            results.get(
                "metadatas",
                [[]]
            )[0]
        )

        distances = (
            results.get(
                "distances",
                [[]]
            )[0]
        )

        chunks = []

        for index, document in enumerate(documents):

            if not document:
                continue

            metadata = (
                metadatas[index]
                if index < len(metadatas)
                else {}
            )

            distance = (
                distances[index]
                if index < len(distances)
                else None
            )

            chunks.append(
                {
                    "content": document,
                    "metadata": metadata,
                    "distance": distance,
                }
            )

        return chunks

    # =========================================================
    # IDENTIFY ALAKART PRODUCT
    # =========================================================

    def _is_alakart_product(
        self,
        metadata: Dict[str, Any]
    ) -> bool:
        """
        Identifies an Alakart product ONLY from trusted metadata.

        Navigation documents are explicitly excluded.
        """

        if not metadata:
            return False

        source = str(
            metadata.get(
                "source",
                ""
            )
        ).strip().lower()

        source_type = str(
            metadata.get(
                "source_type",
                ""
            )
        ).strip().lower()

        document_type = str(
            metadata.get(
                "document_type",
                ""
            )
        ).strip().lower()

        category = str(
            metadata.get(
                "category",
                ""
            )
        ).strip().lower()

        # -----------------------------------------------------
        # Navigation is NEVER a product
        # -----------------------------------------------------

        if category == "navigation":
            return False

        if "navigation" in source:
            return False

        # -----------------------------------------------------
        # Trusted Alakart product metadata
        # -----------------------------------------------------

        return (
            source == "alakart_catalogue"
            or source_type == "alakart_catalogue"
            or document_type == "product_catalog"
            or category == "products"
        )

    # =========================================================
    # MERGE UNIQUE RESULTS
    # =========================================================

    def _merge_unique_chunks(
        self,
        general_chunks: List[Dict[str, Any]],
        alakart_chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Combines results while avoiding duplicate chunks.
        """

        result = []

        seen = set()

        for chunk in (
            general_chunks +
            alakart_chunks
        ):

            metadata = chunk.get(
                "metadata",
                {}
            )

            chunk_id = metadata.get(
                "chunk_id"
            )

            # Use chunk_id when available.
            if chunk_id:

                unique_key = str(
                    chunk_id
                )

            else:

                # Fallback to content.
                unique_key = chunk.get(
                    "content",
                    ""
                ).strip()

            if not unique_key:
                continue

            if unique_key in seen:
                continue

            seen.add(unique_key)

            result.append(chunk)

        return result