def split_text(text, chunk_size=500, overlap=50):
    """
    Split text into smaller chunks with overlap.
    """

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def create_chunks(documents, chunk_size=500, overlap=50):
    """
    Create chunks while preserving source filename
    and page number.
    """

    all_chunks = []

    for doc in documents:

        text_chunks = split_text(
            doc["text"],
            chunk_size=chunk_size,
            overlap=overlap
        )

        for chunk in text_chunks:

            all_chunks.append({
                "text": chunk,
                "source": doc["source"],
                "page": doc["page"]
            })

    return all_chunks