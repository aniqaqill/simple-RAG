import pytest
from unittest.mock import MagicMock, patch
from src.vector_store import VectorStore

class TestVectorStore:
    @pytest.fixture
    def mock_qdrant(self):
        with patch("src.vector_store.QdrantClient") as MockClient:
            yield MockClient

    def test_init_creates_collection_if_missing(self, mock_qdrant):
        # Setup mock to return empty collections list (simulating missing)
        mock_instance = mock_qdrant.return_value
        mock_instance.get_collections.return_value.collections = []
        
        VectorStore()
        
        # Verify creating collection was called
        mock_instance.create_collection.assert_called_once()

    def test_add_upserts_points(self, mock_qdrant):
        vs = VectorStore()
        mock_instance = mock_qdrant.return_value
        
        chunk = "test chunk"
        embedding = [0.1, 0.2, 0.3]
        
        vs.add(chunk, embedding)
        
        # Verify upsert called
        mock_instance.upsert.assert_called_once()
        call_args = mock_instance.upsert.call_args
        assert call_args.kwargs['collection_name'] == vs.collection_name
        assert len(call_args.kwargs['points']) == 1
        assert call_args.kwargs['points'][0].vector == embedding
        assert call_args.kwargs['points'][0].payload['text'] == chunk

    def test_search_returns_results(self, mock_qdrant):
        vs = VectorStore()
        mock_instance = mock_qdrant.return_value
        
        # Mock search response
        mock_point = MagicMock()
        mock_point.payload = {"text": "found chunk"}
        mock_point.score = 0.95
        mock_instance.search.return_value = [mock_point]
        
        results = vs.search([0.1, 0.2, 0.3])
        
        assert len(results) == 1
        assert results[0][0] == "found chunk"
        assert results[0][1] == 0.95
