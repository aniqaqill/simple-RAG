import uuid
from typing import List, Tuple
from qdrant_client import QdrantClient
from qdrant_client.http import models
from .config import QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME

class VectorStore:
    def __init__(self):
        print(f"Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}...")
        self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        self.collection_name = COLLECTION_NAME
        self._ensure_collection_exists()

    def _ensure_collection_exists(self):
        """Checks if collection exists, if not creates it."""
        try:
            collections_response = self.client.get_collections()
            collection_names = [c.name for c in collections_response.collections]
            
            if self.collection_name not in collection_names:
                print(f"Collection '{self.collection_name}' not found. Creating it...")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE),
                )
            else:
                print(f"Collection '{self.collection_name}' exists.")
        except Exception as e:
            print(f"Error checking/creating collection: {e}")
            raise
    def add(self, chunk: str, embedding: List[float]):
        """Adds a chunk and its embedding to Qdrant."""
        point_id = str(uuid.uuid4())
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={"text": chunk}
                )
            ]
        )

    def search(self, query_embedding: List[float], top_n: int = 3) -> List[Tuple[str, float]]:
        """
        Retrieves the top_n most similar chunks to the query embedding.
        """
        try:
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_n
            )

            results = []
            for scored_point in search_result:
                chunk = scored_point.payload.get("text", "") if scored_point.payload else ""
                results.append((chunk, scored_point.score))

            return results
        except Exception as e:
            print(f"Error performing search in Qdrant: {e}")
            raise
    
    def __len__(self):
        # Approximate count
        info = self.client.get_collection(self.collection_name)
        return info.points_count
