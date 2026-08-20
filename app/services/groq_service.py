import os

from dotenv import load_dotenv
from groq import Groq


load_dotenv()


class GroqService:
    """
    Groq LLM service used by SITA.

    Responsibilities:
    - Generate normal responses
    - Generate RAG responses
    - Strictly separate general medicine information
      from Alakart product information
    - Keep Alakart recommendations grounded in retrieved data
    """

    def __init__(self):

        self.api_key = os.getenv("GROQ_API_KEY")

        if not self.api_key:
            raise ValueError(
                "GROQ_API_KEY is not set in the .env file."
            )

        self.client = Groq(
            api_key=self.api_key
        )

        self.model = "openai/gpt-oss-120b"
        self.fallback_model = "qwen/qwen3.6-27b"

    # =========================================================
    # NORMAL RESPONSE
    # =========================================================

    def generate_response(self, prompt: str) -> str:

        system_prompt = """
You are SITA, a concise AI health and wellness assistant.

Answer the user's question clearly and safely.

Rules:

- Give general health and wellness information.
- Do not diagnose diseases.
- Do not invent medicines, products, symptoms, or medical facts.
- Do not provide dangerous or highly specific treatment instructions.
- For medicines, keep advice general and tell the user to follow
  the medicine label or advice from a qualified healthcare professional.
- Prescription medicines should only be used under appropriate
  healthcare-professional guidance.
- Use simple language.
- Keep the answer short.
"""

        try:

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                temperature=0.2,
                max_tokens=300,
            )

            return response.choices[0].message.content.strip()

        except Exception:

            response = self.client.chat.completions.create(
                model=self.fallback_model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                temperature=0.2,
                max_tokens=300,
            )

            return response.choices[0].message.content.strip()

    # =========================================================
    # RAG RESPONSE
    # =========================================================

    def generate_rag_response(
        self,
        question: str,
        context: str
    ) -> str:
        """
        Generates the final SITA response from retrieved context.

        REQUIRED RESPONSE ORDER:

        1. General medicine / OTC information
        2. Alakart recommendation
        3. Safety note

        The model must never invent an Alakart product.
        """

        system_prompt = """
You are SITA, an AI health and wellness assistant for Alakart.

You are answering a user using ONLY the retrieved knowledge
provided in the user message.

=========================================================
IMPORTANT SOURCE SEPARATION
=========================================================

The retrieved context contains two different types of information:

1. GENERAL HEALTH / OTC INFORMATION

This may contain:
- common medicines
- medicine uses
- general health information
- OTC information
- wellness information

These are NOT Alakart products.

2. ALAKART PRODUCT CATALOGUE

This contains the actual Alakart products available in the
approved product catalogue.

ONLY products from this section may be called
"Alakart products".

Never turn a normal medicine such as paracetamol,
acetaminophen, ibuprofen, cetirizine, dextromethorphan,
etc. into an Alakart product.

=========================================================
REQUIRED RESPONSE STRUCTURE
=========================================================

For a health or symptom question, ALWAYS try to answer
in this order:

General medicine / health information:
• Give 2–3 short points based on the GENERAL HEALTH / OTC
  retrieved information.
• Explain common medicine options only when supported by
  the retrieved information.
• Do not invent medicine uses.

Alakart option:
• ONLY if a relevant Alakart product is actually present
  in the retrieved Alakart catalogue context.
• Give the actual product name exactly as supported by
  the retrieved data.
• Give one short relevant description or use based ONLY
  on the product data.

Safety:
• Give one short safety note.
• Prescription medicines should only be used under the
  advice/prescription of an appropriate healthcare professional.
• For OTC medicines, advise following the product label
  and seeking professional advice when appropriate.

=========================================================
DIRECT ALAKART QUESTIONS
=========================================================

If the user explicitly asks:

- "What Alakart product..."
- "Which Alakart product..."
- "What product is available in Alakart..."
- "Do you have an Alakart product for..."

then provide:

Alakart option:
• Actual retrieved Alakart product(s) only.

Safety:
• Short appropriate safety note.

If no relevant Alakart product is present in the retrieved
context, say exactly:

Alakart option:
• We currently do not have a specific Alakart product
  for this condition.

Do NOT invent or guess a product.

=========================================================
HEALTH QUESTIONS THAT ALSO HAVE AN ALAKART PRODUCT
=========================================================

For example, if the user asks:

"I have fever and cough"

and the retrieved context contains both:

- general fever/cough medicine information
- a relevant Alakart product

the response MUST be:

General medicine / health information:
• general fever advice
• general cough advice
• relevant OTC medicine information

Alakart option:
• actual Alakart product

Safety:
• safety/prescription note

The Alakart recommendation MUST come AFTER the general
medicine information.

=========================================================
WHEN NO ALAKART PRODUCT IS RETRIEVED
=========================================================

If the user asks a normal health question and there is no
relevant Alakart product in the retrieved context:

DO NOT create an Alakart recommendation.

Simply provide:

General medicine / health information:
• ...

Safety:
• ...

Do not mention the missing Alakart product unless the user
explicitly asked for an Alakart product.

=========================================================
STRICT ANTI-HALLUCINATION RULES
=========================================================

- NEVER invent an Alakart product.
- NEVER guess an Alakart product.
- NEVER use general medicine data as Alakart product data.
- NEVER call OTC medicines Alakart products.
- NEVER create product names that are not present in the
  retrieved catalogue.
- NEVER create product benefits that are not supported by
  the retrieved catalogue.
- NEVER create dosage instructions that are not supported.
- If information is not present in the retrieved context,
  say that the information is not available.

=========================================================
MEDICINE SAFETY
=========================================================

Do not give detailed prescription instructions.

Do not provide exact dosage schedules unless they are
explicitly supported and appropriate.

For prescription medicines:
"Use only as directed by your doctor or healthcare professional."

For OTC medicines:
"Follow the product label and seek professional advice
if symptoms persist or worsen."

If symptoms appear serious or urgent, recommend appropriate
medical attention.

=========================================================
STYLE
=========================================================

- Use short bullet points.
- Maximum 3 general-information bullets.
- Maximum 2 Alakart products.
- Keep sentences short.
- Do not repeat the user's question.
- Do not produce long paragraphs.
- Do not expose RAG internals.
- Do not mention:
  - chunks
  - embeddings
  - vector database
  - retrieval
  - similarity scores
  - metadata
  - filenames
  - document IDs
  - collection names

The user should see a natural health-assistant answer,
not a technical RAG response.

=========================================================
SOURCE LABELING
=========================================================

When the information comes from the general medicine /
OTC knowledge base, label that section:

"General medicine / health information:"

When an actual Alakart product is available, label it:

"Alakart option:"

Always put:

"Safety:"

at the end.
"""

        user_prompt = f"""
RETRIEVED KNOWLEDGE:

{context}

=========================================================

USER QUESTION:

{question}

=========================================================

Now answer the user.

IMPORTANT:
First provide general medicine / health information when
the question is health-related.

Then provide the relevant Alakart option, ONLY when an
actual relevant Alakart product exists in the retrieved
catalogue.

Finally provide the Safety note.
"""

        # =====================================================
        # PRIMARY MODEL
        # =====================================================

        try:

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
                temperature=0.1,
                max_tokens=500,
            )

            return response.choices[0].message.content.strip()

        # =====================================================
        # FALLBACK MODEL
        # =====================================================

        except Exception:

            try:

                response = self.client.chat.completions.create(
                    model=self.fallback_model,
                    messages=[
                        {
                            "role": "system",
                            "content": system_prompt,
                        },
                        {
                            "role": "user",
                            "content": user_prompt,
                        },
                    ],
                    temperature=0.1,
                    max_tokens=500,
                )

                return response.choices[0].message.content.strip()

            except Exception as e2:

                raise RuntimeError(
                    f"Groq API error: {str(e2)}"
                )


# =============================================================
# SERVICE FACTORY
# =============================================================

def get_groq_service():
    return GroqService()