import faiss
import numpy as np


def create_vector_store(embeddings):
    """
    Build an in-memory FAISS vector index from embeddings.
    Dynamically sizes dimension based on embedding model.
    """
    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )

    if embeddings.size == 0 or embeddings.ndim < 2:
        raise ValueError("Cannot create vector store from empty embeddings.")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return index