import os
import sys
import time
from fastapi.testclient import TestClient

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from app.main import app
from app.services.rag_service import get_rag_service

def test_live_chat_api():
    client = TestClient(app)
    rag = get_rag_service()
    
    queries = [
        "I have a cough. What Alakart product can help?",
        "I have fever and cough.",
        "What is COVI RB01?",
        "Tell me about Herbal Inhaler.",
        "What Alakart product is related to respiratory wellness?",
        "What medicine can help with fever?"
    ]
    
    print("\n" + "=" * 65)
    print("      LIVE /chat API & RAG FLOW VALIDATION TESTS")
    print("=" * 65)
    
    for i, query in enumerate(queries, 1):
        print(f"\n{'='*20} TEST CASE {i}/6: \"{query}\" {'='*20}")
        
        # 1. Inspect RAG pipeline internals (0 LLM tokens consumed)
        retrieved_chunks = rag.retriever.retrieve(query)
        
        alakart_products = []
        for chunk in retrieved_chunks:
            meta = chunk.get("metadata", {})
            if meta.get("source") == "alakart_catalogue" or meta.get("document_type") == "product_catalog":
                name = meta.get("product_name")
                if name and name not in alakart_products:
                    alakart_products.append(name)
        
        prompt_context = rag._build_context(retrieved_chunks)
        
        print("\n--- [INTERNAL LOG: RETRIEVED ALAKART PRODUCTS] ---")
        print(f"Products Found in RAG: {alakart_products if alakart_products else 'None'}")
        
        print("\n--- [INTERNAL LOG: FINAL PROMPT CONTEXT PASSED TO LLM] ---")
        print(prompt_context[:300] + ("..." if len(prompt_context) > 300 else ""))
        
        # 2. Call LIVE /chat endpoint (Testing exact frontend communication layer)
        response = client.post("/chat", json={"message": query})
        
        if response.status_code != 200:
            print(f"\n[ERROR] /chat endpoint returned status {response.status_code}: {response.text}")
        else:
            chat_data = response.json()
            clean_response = chat_data.get("response", "")
            
            print("\n--- [LIVE /chat API FINAL RESPONSE (Sent to Frontend)] ---")
            print(clean_response)
            
            # Assertions to ensure no RAG internals leak to the user
            assert "chunk_" not in clean_response, "Internal chunk ID leaked!"
            assert "vector" not in clean_response.lower(), "RAG vector term leaked!"
            assert "score" not in clean_response.lower(), "Similarity score leaked!"
            
        print("\n" + "-" * 65)
        time.sleep(5)  # Buffer between calls to stay well within rate limits

if __name__ == "__main__":
    test_live_chat_api()
