from typing import Any, Dict, List

from app.services.retriever import Retriever
from app.services.groq_service import get_groq_service


class RAGService:
    """
    Main RAG orchestration layer for SITA.

    Flow:

        User Question
              ↓
        Intent Detection
              ↓
        Retriever
              ↓
        General Medicine Information
              +
        Alakart Product Information
              ↓
        Structured Context
              ↓
        Groq
              ↓
        Final SITA Response
    """

    def __init__(self):
        self.retriever = Retriever(top_k=8)
        self.llm_service = get_groq_service()

    # =========================================================
    # MAIN CHAT PIPELINE
    # =========================================================

    def handle_query(self, query: str) -> str:

        query = query.strip()

        if not query:
            return "Please enter a health or wellness question."

        print("\n" + "=" * 70)
        print("SITA RAG PIPELINE")
        print("=" * 70)

        print(f"USER QUERY:\n{query}")

        # -----------------------------------------------------
        # STEP 1: INTENT
        # -----------------------------------------------------

        intent = self._detect_intent(query)

        print("\n1. DETECTED INTENT")
        print(intent)

        # -----------------------------------------------------
        # STEP 2: RETRIEVE
        # -----------------------------------------------------

        chunks = self.retriever.retrieve(
            query=query,
            intent=intent
        )

        print(
            f"\n2. TOTAL RETRIEVED CHUNKS: {len(chunks)}"
        )

        # -----------------------------------------------------
        # STEP 3: BUILD CONTEXT
        # -----------------------------------------------------

        context = self._build_context(
            chunks=chunks,
            intent=intent
        )

        print("\n3. STRUCTURED CONTEXT")
        print(context)

        # -----------------------------------------------------
        # STEP 4: GROQ
        # -----------------------------------------------------

        response = self.llm_service.generate_rag_response(
            question=query,
            context=context
        )

        print("\n4. FINAL SITA RESPONSE")
        print(response)

        print("=" * 70 + "\n")

        return response

    # =========================================================
    # DEBUG PIPELINE
    # =========================================================

    def handle_query_debug(
        self,
        query: str
    ) -> Dict[str, Any]:

        query = query.strip()

        intent = self._detect_intent(query)

        chunks = self.retriever.retrieve(
            query=query,
            intent=intent
        )

        context = self._build_context(
            chunks=chunks,
            intent=intent
        )

        response = self.llm_service.generate_rag_response(
            question=query,
            context=context
        )

        return {
            "question": query,
            "intent": intent,
            "retrieved_chunks": chunks,
            "context": context,
            "response": response,
        }

    # =========================================================
    # INTENT DETECTION
    # =========================================================

    def _detect_intent(
        self,
        query: str
    ) -> Dict[str, Any]:

        text = query.lower().strip()

        # -----------------------------------------------------
        # ALAKART
        # -----------------------------------------------------

        alakart_keywords = [
            "alakart",
            "ala kart",
            "ala-kart",
        ]

        asks_alakart = any(
            keyword in text
            for keyword in alakart_keywords
        )

        # -----------------------------------------------------
        # PRODUCT
        # -----------------------------------------------------

        product_keywords = [
            "product",
            "products",
            "which product",
            "what product",
            "recommend a product",
            "recommend product",
            "suggest a product",
            "suggest product",
            "available product",
        ]

        asks_product = any(
            keyword in text
            for keyword in product_keywords
        )

        # -----------------------------------------------------
        # GENERAL MEDICINE
        # -----------------------------------------------------

        medicine_keywords = [
            "medicine",
            "medication",
            "tablet",
            "tablets",
            "capsule",
            "capsules",
            "syrup",
            "drug",
            "drugs",
            "otc",
            "over the counter",
            "what can i take",
            "what should i take",
        ]

        asks_general_medicine = any(
            keyword in text
            for keyword in medicine_keywords
        )

        # -----------------------------------------------------
        # HEALTH
        # -----------------------------------------------------

        health_keywords = [
            "fever",
            "cold",
            "cough",
            "headache",
            "pain",
            "sore throat",
            "throat pain",
            "body pain",
            "runny nose",
            "blocked nose",
            "congestion",
            "allergy",
            "allergies",
            "sneeze",
            "sneezing",
            "vomiting",
            "nausea",
            "diarrhea",
            "diarrhoea",
            "stomach",
            "indigestion",
            "acidity",
            "gas",
            "fatigue",
            "weakness",
            "sleep",
            "stress",
            "wellness",
            "symptom",
            "symptoms",
        ]

        asks_health = any(
            keyword in text
            for keyword in health_keywords
        )

        if asks_alakart:
            asks_health = True

        if asks_general_medicine:
            asks_health = True

        return {
            "asks_alakart": asks_alakart,
            "asks_general_medicine": asks_general_medicine,
            "asks_health": asks_health,
            "asks_product": asks_product,
        }

    # =========================================================
    # BUILD STRUCTURED CONTEXT
    # =========================================================

    def _build_context(
        self,
        chunks: List[Dict[str, Any]],
        intent: Dict[str, Any]
    ) -> str:

        general_chunks = []
        alakart_chunks = []

        for chunk in chunks:

            content = str(
                chunk.get("content", "")
            ).strip()

            if not content:
                continue

            metadata = chunk.get(
                "metadata",
                {}
            )

            if self._is_alakart_product(metadata):
                alakart_chunks.append(chunk)
            else:
                general_chunks.append(chunk)

        # -----------------------------------------------------
        # LIMIT EACH SECTION
        # -----------------------------------------------------

        general_chunks = general_chunks[:5]
        alakart_chunks = alakart_chunks[:3]

        # -----------------------------------------------------
        # BUILD SECTIONS
        # -----------------------------------------------------

        general_section = self._format_general_chunks(
            general_chunks
        )

        alakart_section = self._format_alakart_chunks(
            alakart_chunks
        )

        intent_section = f"""
### USER INTENT

asks_health: {intent.get("asks_health", False)}
asks_general_medicine: {intent.get("asks_general_medicine", False)}
asks_alakart: {intent.get("asks_alakart", False)}
asks_product: {intent.get("asks_product", False)}
""".strip()

        return (
            intent_section
            + "\n\n"
            + general_section
            + "\n\n"
            + "=================================================="
            + "\n\n"
            + alakart_section
        )

    # =========================================================
    # GENERAL INFORMATION
    # =========================================================

    def _format_general_chunks(
        self,
        chunks: List[Dict[str, Any]]
    ) -> str:

        if not chunks:
            return (
                "### GENERAL HEALTH / OTC / WELLNESS INFORMATION\n"
                "[No relevant general information retrieved.]"
            )

        parts = [
            "### GENERAL HEALTH / OTC / WELLNESS INFORMATION",
            "",
            "This section contains general medicine, OTC, "
            "health and wellness information.",
            "",
        ]

        for index, chunk in enumerate(
            chunks,
            start=1
        ):

            metadata = chunk.get(
                "metadata",
                {}
            )

            category = str(
                metadata.get(
                    "category",
                    "general"
                )
            )

            source = str(
                metadata.get(
                    "source",
                    "general_health"
                )
            )

            content = str(
                chunk.get(
                    "content",
                    ""
                )
            ).strip()

            parts.append(
                f"[GENERAL SOURCE {index}]"
            )

            parts.append(
                f"Category: {category}"
            )

            parts.append(
                f"Source: {source}"
            )

            parts.append(content)
            parts.append("")

        return "\n".join(parts)

    # =========================================================
    # ALAKART INFORMATION
    # =========================================================

    def _format_alakart_chunks(
        self,
        chunks: List[Dict[str, Any]]
    ) -> str:

        if not chunks:
            return (
                "### ALAKART PRODUCT INFORMATION\n"
                "[No relevant Alakart product was retrieved.]"
            )

        parts = [
            "### ALAKART PRODUCT INFORMATION",
            "",
            "IMPORTANT:",
            "Only products in this section are approved "
            "Alakart products.",
            "Never convert a general medicine into an "
            "Alakart product.",
            "",
        ]

        for index, chunk in enumerate(
            chunks,
            start=1
        ):

            metadata = chunk.get(
                "metadata",
                {}
            )

            product_name = str(
                metadata.get(
                    "product_name",
                    ""
                )
            ).strip()

            category = str(
                metadata.get(
                    "category",
                    "products"
                )
            )

            source = str(
                metadata.get(
                    "source",
                    "alakart_catalogue"
                )
            )

            content = str(
                chunk.get(
                    "content",
                    ""
                )
            ).strip()

            parts.append(
                f"[ALAKART PRODUCT {index}]"
            )

            if product_name:
                parts.append(
                    f"Product Name: {product_name}"
                )

            parts.append(
                f"Category: {category}"
            )

            parts.append(
                f"Source: {source}"
            )

            parts.append(content)
            parts.append("")

        return "\n".join(parts)

    # =========================================================
    # PRODUCT IDENTIFICATION
    # =========================================================

    def _is_alakart_product(
        self,
        metadata: Dict[str, Any]
    ) -> bool:

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

        # Navigation is NEVER an Alakart product.
        if category == "navigation":
            return False

        if "navigation" in source:
            return False

        return (
            source == "alakart_catalogue"
            or source_type == "alakart_catalogue"
            or document_type == "product_catalog"
            or category == "products"
        )


# =============================================================
# FACTORY
# =============================================================

def get_rag_service():
    return RAGService()