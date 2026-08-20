from typing import List, Dict, Any, ClassVar

from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]

from app.services.document_loader import Document


class EmbeddingService:
    """
    Generates sentence embeddings for Document chunks using a local
    Hugging Face sentence-transformers model.

    Model:
        all-MiniLM-L6-v2

    - Lightweight (~80 MB)
    - Fast
    - No API key required
    - Produces 384-dimensional embeddings
    - Good general-purpose performance for English text

    Performance:
        The SentenceTransformer model is cached at the class level so
        multiple EmbeddingService instances reuse the same loaded model.
        This avoids repeatedly loading the model during application
        requests.
    """

    DEFAULT_MODEL = "all-MiniLM-L6-v2"

    # Cache loaded models by model name.
    # This prevents repeated model initialization.
    _model_cache: ClassVar[Dict[str, SentenceTransformer]] = {}

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL
    ):
        self.model_name = model_name

        # Reuse an already-loaded model.
        if model_name not in self._model_cache:
            print(
                f"Loading embedding model: {model_name}"
            )

            self._model_cache[model_name] = (
                SentenceTransformer(model_name)
            )

            print(
                f"Embedding model loaded: {model_name}"
            )

        self._model = self._model_cache[model_name]

    # =========================================================
    # EMBEDDING DIMENSION
    # =========================================================

    @property
    def embedding_dimension(self) -> int:
        """
        Returns the output vector dimensionality of the
        loaded embedding model.
        """

        return self._model.get_sentence_embedding_dimension()

    # =========================================================
    # SINGLE TEXT EMBEDDING
    # =========================================================

    def embed_text(
        self,
        text: str
    ) -> List[float]:
        """
        Generates an embedding for a single text string.

        Returns:
            List[float]
        """

        if not text or not text.strip():
            return []

        vector = self._model.encode(
            text,
            convert_to_numpy=True,
            show_progress_bar=False
        )

        return vector.tolist()

    # =========================================================
    # DOCUMENT EMBEDDING
    # =========================================================

    def embed_documents(
        self,
        documents: List[Document]
    ) -> List[Dict[str, Any]]:
        """
        Generates embeddings for a list of Document chunks.

        Returns a list of dictionaries containing:

            embedding:
                List[float]

            page_content:
                Original document/chunk text

            metadata:
                Original document metadata
        """

        if not documents:
            return []

        texts = [
            doc.page_content
            for doc in documents
        ]

        vectors = self._model.encode(
            texts,
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        results = []

        for doc, vector in zip(
            documents,
            vectors
        ):
            results.append(
                {
                    "embedding": vector.tolist(),
                    "page_content": doc.page_content,
                    "metadata": doc.metadata,
                }
            )

        return results