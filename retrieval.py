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

    if not chunks or vector_store is None:
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

    # Search FAISS for similar chunks
    distances, indices = vector_store.search(
        question_embedding,
        min(top_k, len(chunks))
    )

    relevant_chunks = []

    for index in indices[0]:

        if index < 0 or index >= len(chunks):
            continue

        # Return the complete dictionary.
        # Do not return only chunk["text"].
        relevant_chunks.append(chunks[index])

    return relevant_chunks