import json
from typing import Any, Dict, List

from app.services.document_loader import Document


class ProductChunker:
    """
    Product-aware chunker for the Alakart product catalogue.

    Responsibilities:
    1. Identify Alakart product documents.
    2. Parse structured product JSON.
    3. Create one chunk per product.
    4. Preserve clear source metadata.
    5. Preserve product-specific metadata.
    6. Prevent duplicate products.
    """

    # =========================================
    # IDENTIFY PRODUCT DOCUMENT
    # =========================================

    def is_product_document(
        self,
        doc: Document
    ) -> bool:

        metadata = doc.metadata

        category = str(
            metadata.get("category", "")
        ).strip().lower()

        source_type = str(
            metadata.get("source_type", "")
        ).strip().lower()

        document_type = str(
            metadata.get("document_type", "")
        ).strip().lower()

        source = str(
            metadata.get("source", "")
        ).strip().lower()

        # -----------------------------------------
        # Preferred metadata classification
        # -----------------------------------------

        if source_type == "alakart_catalogue":
            return True

        if document_type == "product_catalog":
            return True

        # -----------------------------------------
        # Backward compatibility
        # -----------------------------------------

        if category == "products":
            return True

        if "product" in source:
            return True

        return False

    # =========================================
    # CHUNK PRODUCT DOCUMENT
    # =========================================

    def chunk_product_document(
        self,
        doc: Document
    ) -> List[Document]:

        text = doc.page_content

        products = self._parse_products(text)

        # -----------------------------------------
        # If JSON could not be parsed
        # -----------------------------------------

        if not products:

            print(
                "WARNING: No structured products "
                "could be parsed from document."
            )

            return [doc]

        chunks: List[Document] = []

        seen_products = set()

        # -----------------------------------------
        # Create one chunk per product
        # -----------------------------------------

        for product in products:

            if not isinstance(product, dict):
                continue

            # -----------------------------------------
            # Product name
            # -----------------------------------------

            name = str(
                product.get(
                    "product_name",
                    ""
                )
            ).strip()

            if not name:
                continue

            normalized_name = name.lower()

            # -----------------------------------------
            # Deduplicate
            # -----------------------------------------

            if normalized_name in seen_products:
                continue

            seen_products.add(normalized_name)

            # -----------------------------------------
            # Format product content
            # -----------------------------------------

            chunk_text = self._format_product_chunk(
                product
            )

            # -----------------------------------------
            # Preserve document metadata
            # -----------------------------------------

            chunk_metadata = doc.metadata.copy()

            # -----------------------------------------
            # STRICT ALAKART SOURCE METADATA
            # -----------------------------------------

            # category identifies the knowledge source.
            chunk_metadata["category"] = "products"

            # source_type identifies this as the
            # Alakart product catalogue.
            chunk_metadata["source_type"] = (
                "alakart_catalogue"
            )

            # document_type identifies the document
            # as product catalogue data.
            chunk_metadata["document_type"] = (
                "product_catalog"
            )

            # -----------------------------------------
            # Preserve source filename
            # -----------------------------------------

            if not chunk_metadata.get("source"):
                chunk_metadata["source"] = (
                    "Alakart Product Catalogue"
                )

            if not chunk_metadata.get("source_name"):
                chunk_metadata["source_name"] = (
                    chunk_metadata["source"]
                )

            # -----------------------------------------
            # Product-specific metadata
            # -----------------------------------------

            chunk_metadata["product_name"] = name

            product_category = product.get(
                "category"
            )

            if product_category:

                chunk_metadata["product_category"] = (
                    str(product_category).strip()
                )

            # -----------------------------------------
            # Create product Document
            # -----------------------------------------

            chunks.append(
                Document(
                    page_content=chunk_text,
                    metadata=chunk_metadata,
                )
            )

        print(
            f"ProductChunker: Created "
            f"{len(chunks)} Alakart product chunks."
        )

        return chunks

    # =========================================
    # PARSE PRODUCT JSON
    # =========================================

    def _parse_products(
        self,
        text: str
    ) -> List[Dict[str, Any]]:

        """
        Parses structured product JSON.

        Supported formats:

        1. Direct array:

        [
            {
                "product_name": "...",
                ...
            }
        ]

        2. Object containing products:

        {
            "products": [
                {
                    "product_name": "...",
                    ...
                }
            ]
        }
        """

        text = text.strip()

        if not text:
            return []

        # =========================================
        # ATTEMPT 1
        # Parse entire document as JSON
        # =========================================

        try:

            data = json.loads(text)

            # -----------------------------------------
            # JSON array
            # -----------------------------------------

            if isinstance(data, list):

                return [
                    item
                    for item in data
                    if isinstance(item, dict)
                ]

            # -----------------------------------------
            # JSON object containing products
            # -----------------------------------------

            if isinstance(data, dict):

                products = data.get("products")

                if isinstance(products, list):

                    return [
                        item
                        for item in products
                        if isinstance(item, dict)
                    ]

        except Exception:
            pass

        # =========================================
        # ATTEMPT 2
        # Find JSON array inside text
        # =========================================

        try:

            start = text.find("[")
            end = text.rfind("]")

            if start != -1 and end != -1:

                json_str = text[
                    start:end + 1
                ]

                data = json.loads(json_str)

                if isinstance(data, list):

                    return [
                        item
                        for item in data
                        if isinstance(item, dict)
                    ]

        except Exception as e:

            print(
                f"Error parsing product JSON: {e}"
            )

        return []

    # =========================================
    # FORMAT PRODUCT CHUNK
    # =========================================

    def _format_product_chunk(
        self,
        product: Dict[str, Any]
    ) -> str:

        parts: List[str] = []

        # =========================================
        # PRODUCT NAME
        # =========================================

        product_name = str(
            product.get(
                "product_name",
                ""
            )
        ).strip()

        if product_name:

            parts.append(
                f"Product Name: {product_name}"
            )

        # =========================================
        # PRODUCT CATEGORY
        # =========================================

        if product.get("category"):

            parts.append(
                f"Product Category: "
                f"{product['category']}"
            )

        # =========================================
        # DESCRIPTION
        # =========================================

        if product.get("description"):

            parts.append(
                f"Description: "
                f"{product['description']}"
            )

        # =========================================
        # OTHER PRODUCT FIELDS
        # =========================================

        excluded_fields = {
            "product_name",
            "category",
            "description",
        }

        for key, value in product.items():

            if key in excluded_fields:
                continue

            if value is None:
                continue

            if value == "":
                continue

            # -----------------------------------------
            # Convert nested JSON structures
            # into readable text
            # -----------------------------------------

            if isinstance(
                value,
                (dict, list)
            ):

                value = json.dumps(
                    value,
                    ensure_ascii=False
                )

            readable_key = (
                str(key)
                .replace("_", " ")
                .strip()
                .title()
            )

            parts.append(
                f"{readable_key}: {value}"
            )

        # =========================================
        # EXPLICIT KNOWLEDGE SOURCE
        # =========================================

        parts.append(
            "Knowledge Source: "
            "Alakart Product Catalogue"
        )

        return "\n".join(parts)