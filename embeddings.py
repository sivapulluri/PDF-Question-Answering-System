import os
import numpy as np
from gemini_llm import get_gemini_client


class GeminiEmbeddingModel:
    """
    Lightweight embedding model wrapper using the Gemini Embedding API.
    Replaces heavy SentenceTransformers/PyTorch, reducing memory footprint
    from >530MB to <100MB and preventing OOM crashes on 512MB hosting (Render).
    """

    def __init__(self, model_name=None):
        self.model_name = model_name or os.getenv(
            "GEMINI_EMBEDDING_MODEL", "gemini-embedding-001"
        )
        self._client = None

    @property
    def client(self):
        if self._client is None:
            self._client = get_gemini_client()
        return self._client

    def encode(self, texts, convert_to_numpy=True, batch_size=50):
        """
        Encode text or list of texts into vector embeddings.
        Compatible with SentenceTransformer.encode() interface.
        """
        if isinstance(texts, str):
            texts = [texts]

        if not texts:
            if convert_to_numpy:
                return np.empty((0, 3072), dtype="float32")
            return []

        all_vectors = []

        # Batch requests to stay within API limits
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = self.client.models.embed_content(
                model=self.model_name,
                contents=batch
            )
            for item in response.embeddings:
                all_vectors.append(item.values)

        if convert_to_numpy:
            return np.asarray(all_vectors, dtype="float32")
        return all_vectors


def create_embeddings(texts, model_name=None):
    """
    Generate vector embeddings for a list of text chunks.
    Returns:
        embeddings (np.ndarray): 2D float32 array of shape (N, dimension)
        model (GeminiEmbeddingModel): Compatible embedding model instance for querying
    """
    model = GeminiEmbeddingModel(model_name=model_name)
    embeddings = model.encode(texts, convert_to_numpy=True)
    return embeddings, model