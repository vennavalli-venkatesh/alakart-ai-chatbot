import os
import sys
import time

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from app.services.rag_service import get_rag_service

def test_queries():
    rag_service = get_rag_service()
    
    queries = [
        "I have fever and cough",
        "I have a cough. What Alakart product can help?",
        "What is COVI RB01?",
        "What medicine can help with fever?",
        "I have an earache and jaw pain"
    ]
    
    print("\n========================================")
    print("--- SITA RAG RESPONSE-GENERATION TESTS ---")
    print("========================================")
    
    for i, query in enumerate(queries, 1):
        print(f"\n[{i}/5] USER QUERY: \"{query}\"")
        print("-" * 50)
        
        try:
            # Also get debug info to show retrieval verification
            debug_info = rag_service.handle_query_debug(query)
            
            # Print retrieved products (if any)
            alakart_retrieved = []
            for chunk in debug_info.get("retrieved_chunks", []):
                meta = chunk.get("metadata", {})
                if meta.get("source") == "alakart_catalogue" or meta.get("document_type") == "product_catalog":
                    alakart_retrieved.append(meta.get("product_name", "Unknown"))
            
            print(f"Retrieved Alakart Products: {list(set(alakart_retrieved)) or 'None'}")
            print(f"Generated SITA Response:\n{debug_info['response']}")
            print("-" * 50)
        except Exception as e:
            print(f"Error processing query: {e}")
            
        time.sleep(3)

if __name__ == "__main__":
    test_queries()
