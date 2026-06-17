from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def create_vectorstore(chunks):
    """Create a FAISS vector store from document chunks."""

    # Step 1: Load the embedding model (runs locally, 100% free)
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
    )

    # Step 2: Create FAISS index from chunks
    vectorstore = FAISS.from_documents(chunks, embeddings)

    return vectorstore


# Quick test
if __name__ == "__main__":
    from ingest import load_and_split_pdf
    import sys

    if len(sys.argv) < 2:
        print("Usage: python vectorstore.py <path_to_pdf>")
        sys.exit(1)

    chunks = load_and_split_pdf(sys.argv[1])
    vectorstore = create_vectorstore(chunks)

    # Test a search query
    query = "What is a dictionary?"
    results = vectorstore.similarity_search(query, k=2)

    print(f"\n--- Top 2 results for: '{query}' ---")
    for i, doc in enumerate(results):
        print(f"\n[Result {i+1}]")
        print(doc.page_content[:300])
