import pytest
import time
from src.rag import RAGPipeline
from src.vector_store import VectorStore

# Mark as integration so it can be skipped or selected specifically
@pytest.mark.integration
def test_full_rag_flow():
    """
    Tests the full flow using the running Qdrant instance and Ollama.
    Requires:
    - Docker container for Qdrant running.
    - Ollama running.
    """
    try:
        pipeline = RAGPipeline()
    except Exception as e:
        pytest.skip(f"Skipping integration test: Could not init pipeline (services might be down): {e}")

    # 1. Index a unique test document
    unique_fact = f"Integration test cat fact {time.time()}: Cyber cats love neon lights."
    pipeline.index_documents([unique_fact])
    
    # 2. Retrieve it
    # Qdrant is near real-time but slight delay sometimes
    time.sleep(1) 
    
    results = pipeline.retrieve("What do cyber cats love?")
    
    # 3. Verify retrieval
    found = False
    for chunk, score in results:
        if "Cyber cats love neon lights" in chunk:
            found = True
            break
    
    assert found, "Failed to retrieve the indexed document from Qdrant"
