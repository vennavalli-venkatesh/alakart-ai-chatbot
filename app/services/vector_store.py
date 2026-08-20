import os
from typing import Any, Dict, List

import chromadb


class VectorStore:
    """
    Manages the persistent ChromaDB vector store.

    The vector store stores:

    - document embeddings
    - original document/chunk text
    - metadata

    Metadata is important because our RAG pipeline needs to
    distinguish between:

        products  -> Alakart catalogue
        otc       -> general medicine information
        wellness  -> wellness information
        navigation -> app/navigation information
    """

    def __init__(
        self,
        persist_directory: str = "chroma_db",
        collection_name: str = "rag_collection",
    ):

        self.persist_directory = persist_directory
        self.collection_name = collection_name

        # =========================================
        # CREATE PERSISTENT DIRECTORY
        # =========================================

        os.makedirs(
            self.persist_directory,
            exist_ok=True,
        )

        # =========================================
        # CREATE CHROMA CLIENT
        # =========================================

        self.client = chromadb.PersistentClient(
            path=self.persist_directory
        )

        # =========================================
        # GET OR CREATE COLLECTION
        # =========================================

        self.collection = (
            self.client.get_or_create_collection(
                name=self.collection_name
            )
        )

    # =========================================
    # RESET COLLECTION
    # =========================================

    def reset_collection(self):
        """
        Deletes the existing collection and creates
        a completely new empty collection.

        Used when rebuilding the RAG index after
        changing the source data or chunking logic.
        """

        try:

            self.client.delete_collection(
                name=self.collection_name
            )

        except Exception:

            # Collection may not exist yet.
            pass

        self.collection = (
            self.client.create_collection(
                name=self.collection_name
            )
        )

    # =========================================
    # ADD DOCUMENTS
    # =========================================

    def add_documents(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
    ):
        """
        Adds document chunks and their embeddings
        to ChromaDB.

        Metadata is preserved so the retrieval layer
        can identify the knowledge source.
        """

        if not ids:
            return

        if not embeddings:
            return

        if not documents:
            return

        if not metadatas:
            return

        # =========================================
        # BASIC LENGTH VALIDATION
        # =========================================

        if not (
            len(ids)
            == len(embeddings)
            == len(documents)
            == len(metadatas)
        ):

            raise ValueError(
                "VectorStore.add_documents(): "
                "ids, embeddings, documents and "
                "metadatas must have the same length."
            )

        # =========================================
        # SANITIZE CHROMA METADATA
        # =========================================
        #
        # Chroma metadata values must be simple
        # scalar values.
        #
        # Allowed:
        #   str
        #   int
        #   float
        #   bool
        #
        # Lists/dictionaries are converted to strings.
        # =========================================

        sanitized_metadatas: List[
            Dict[str, Any]
        ] = []

        for metadata in metadatas:

            sanitized: Dict[str, Any] = {}

            for key, value in metadata.items():

                if value is None:
                    continue

                if isinstance(
                    value,
                    (str, int, float, bool),
                ):

                    sanitized[key] = value

                else:

                    sanitized[key] = str(value)

            sanitized_metadatas.append(
                sanitized
            )

        # =========================================
        # ADD TO CHROMA
        # =========================================

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=sanitized_metadatas,
        )

    # =========================================
    # SEARCH
    # =========================================

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 4,
    ) -> dict:
        """
        Searches ChromaDB using the user's query
        embedding.

        Returns:

        {
            "ids": [...],
            "documents": [...],
            "metadatas": [...],
            "distances": [...]
        }
        """

        if not query_embedding:
            return {
                "ids": [[]],
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]],
            }

        # =========================================
        # PROTECT AGAINST INVALID TOP_K
        # =========================================

        collection_count = self.collection.count()

        if collection_count == 0:

            return {
                "ids": [[]],
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]],
            }

        safe_top_k = min(
            max(1, top_k),
            collection_count,
        )

        # =========================================
        # CHROMA SEARCH
        # =========================================

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=safe_top_k,
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

        return results

    # =========================================
    # COLLECTION COUNT
    # =========================================

    def get_collection_count(self) -> int:
        """
        Returns the number of stored chunks.
        """

        return self.collection.count()

    # =========================================
    # COLLECTION INFORMATION
    # =========================================

    def get_collection(self):
        """
        Returns the underlying Chroma collection.

        Useful for developer/debugging tools.
        """

        return self.collection