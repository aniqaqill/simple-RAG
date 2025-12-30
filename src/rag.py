import ollama
from typing import List, Tuple, Generator

from .config import OLLAMA_BASE_URL, EMBEDDING_MODEL, LANGUAGE_MODEL
from .vector_store import VectorStore

class RAGPipeline:
    def __init__(self):
        print(f"Connecting to Windows Ollama at: {OLLAMA_BASE_URL}")
        self.client = ollama.Client(host=OLLAMA_BASE_URL)
        self.vector_store = VectorStore()

    def embed_text(self, text: str) -> List[float]:
        """Embeds text using the configured embedding model."""
        response = self.client.embed(model=EMBEDDING_MODEL, input=text)
        return response['embeddings'][0]

    def index_documents(self, documents: List[str]):
        """Indexes a list of documents into the vector store."""
        print("Indexing documents into Vector DB...")
        count = len(documents)
        for i, chunk in enumerate(documents):
            embedding = self.embed_text(chunk)
            self.vector_store.add(chunk, embedding)
            if (i + 1) % 5 == 0:
                print(f'Added {i+1}/{count} chunks...')

    def retrieve(self, query: str, top_n: int = 3) -> List[Tuple[str, float]]:
        """Retrieves relevant chunks for a query."""
        query_embedding = self.embed_text(query)
        return self.vector_store.search(query_embedding, top_n=top_n)

    def generate_response(self, query: str, context_chunks: List[Tuple[str, float]]) -> Generator:
        """Generates a response using the LLM based on context."""
        
        # Build the context-aware prompt
        context_text = '\n'.join([f' - {chunk}' for chunk, sim in context_chunks])
        instruction_prompt = f'''You are a helpful chatbot. 
Use only the following context to answer. If the answer isn't here, say you don't know.
Context:
{context_text}'''

        return self.client.chat(
            model=LANGUAGE_MODEL,
            messages=[
                {'role': 'system', 'content': instruction_prompt},
                {'role': 'user', 'content': query},
            ],
            stream=True,
        )
