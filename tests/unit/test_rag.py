import pytest
from unittest.mock import MagicMock, patch
from src.rag import RAGPipeline

class TestRAGPipeline:
    @pytest.fixture
    def mock_deps(self):
        with patch("src.rag.ollama.Client") as MockOllama, \
             patch("src.rag.VectorStore") as MockVectorStore:
            yield MockOllama, MockVectorStore

    def test_embed_text(self, mock_deps):
        MockOllama, _ = mock_deps
        mock_client = MockOllama.return_value
        mock_client.embed.return_value = {'embeddings': [[0.1, 0.2, 0.3]]}
        
        rag = RAGPipeline()
        embedding = rag.embed_text("test")
        
        assert embedding == [0.1, 0.2, 0.3]
        mock_client.embed.assert_called_once()

    def test_index_documents(self, mock_deps):
        MockOllama, MockVectorStore = mock_deps
        mock_client = MockOllama.return_value
        mock_vs = MockVectorStore.return_value
        
        # Mock embedding response
        mock_client.embed.return_value = {'embeddings': [[0.1, 0.2]]}
        
        rag = RAGPipeline()
        docs = ["doc1", "doc2"]
        rag.index_documents(docs)
        
        # Both docs should be embedded and added
        assert mock_client.embed.call_count == 2
        assert mock_vs.add.call_count == 2

    def test_retrieve(self, mock_deps):
        MockOllama, MockVectorStore = mock_deps
        mock_client = MockOllama.return_value
        mock_vs = MockVectorStore.return_value
        
        mock_client.embed.return_value = {'embeddings': [[0.1]]}
        mock_vs.search.return_value = [("result", 0.9)]
        
        rag = RAGPipeline()
        results = rag.retrieve("query")
        
        assert len(results) == 1
        assert results[0][0] == "result"
        
    def test_generate_response(self, mock_deps):
        MockOllama, _ = mock_deps
        mock_client = MockOllama.return_value
        
        rag = RAGPipeline()
        context = [("fact1", 0.9)]
        rag.generate_response("query", context)
        
        mock_client.chat.assert_called_once()
        call_args = mock_client.chat.call_args
        messages = call_args.kwargs['messages']
        
        # Check if context is in system prompt
        assert "fact1" in messages[0]['content']
        assert "query" == messages[1]['content']
