import numpy as np


def retrieve_relevant_chunks(
    question,
    embedding_model,
    vector_store,
    chunks,
    top_k=3
):
    """
    Retrieve relevant chunks and preserve source metadata.
    """
    if not chunks or vector_store is None or embedding_model is None:
        return []

    # Create embedding for the user's question
    question_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True
    )

    question_embedding = np.asarray(
        question_embedding,
        dtype="float32"
    )

    # Ensure 2D shape (1, dimension)
    if question_embedding.ndim == 1:
        question_embedding = np.expand_dims(question_embedding, axis=0)

    # Search FAISS for similar chunks
    search_k = max(1, min(top_k, len(chunks)))
    distances, indices = vector_store.search(
        question_embedding,
        search_k
    )

    relevant_chunks = []

    for index in indices[0]:
        if index < 0 or index >= len(chunks):
            continue
        relevant_chunks.append(chunks[index])

    return relevant_chunks


# Alias for backwards compatibility with test scripts
def retrieve_context(query, model, index, all_chunks, top_k=3):
    return retrieve_relevant_chunks(
        question=query,
        embedding_model=model,
        vector_store=index,
        chunks=all_chunks,
        top_k=top_k
    )