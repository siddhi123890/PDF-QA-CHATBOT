from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_and_split_pdf(pdf_path: str):
    """Load a PDF file and split it into smaller text chunks."""

    # Step 1: Load the PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Step 2: Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = splitter.split_documents(documents)

    return chunks


# Quick test
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python ingest.py <path_to_pdf>")
        sys.exit(1)

    chunks = load_and_split_pdf(sys.argv[1])

    # Preview first chunk
    print("\n--- First Chunk Preview ---")
    print(chunks[0].page_content[:500])
