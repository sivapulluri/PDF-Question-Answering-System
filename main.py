from pdf_loader import load_pdfs

from chunking import create_chunks

from embeddings import create_embeddings

from vector_store import create_vector_store

from retrieval import retrieve_context

from gemini_llm import generate_answer


# Step 1: Load PDFs

documents = load_pdfs("documentsss")

print("Total PDFs:", len(documents))


# Step 2: Create chunks

all_chunks = create_chunks(documents)

print("Total chunks:", len(all_chunks))


# Step 3: Create embeddings

texts = [
    chunk["text"]
    for chunk in all_chunks
]

embeddings, embedding_model = create_embeddings(
    texts
)


# Step 4: Create FAISS index

index = create_vector_store(
    embeddings
)


# Step 5: User query

query = input(
    "\nAsk a question about your PDFs: "
)


# Step 6: Retrieve relevant chunks

results = retrieve_context(
    query=query,
    model=embedding_model,
    index=index,
    all_chunks=all_chunks,
    top_k=3
)


# Step 7: Prepare context

context = "\n\n".join(
    [
        f"Source: {r['source']}\n{r['text']}"
        for r in results
    ]
)


# Step 8: Generate answer

answer = generate_answer(
    query=query,
    context=context
)


print("\nAnswer:\n")

print(answer)


# Step 9: Display sources

print("\nSources:")

for result in results:

    print("-", result["source"])