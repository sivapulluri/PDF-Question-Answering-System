from sentence_transformers import SentenceTransformer


def create_embeddings(texts):

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    embeddings = model.encode(
        texts,
        convert_to_numpy=True
    )

    return embeddings, model