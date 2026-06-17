from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import numpy as np

# Load API key from .env file
load_dotenv()

# The refusal message when the answer is not in the PDF
NOT_IN_PDF = "This information is not available in the uploaded PDF. I can only answer questions based on the content of the document you provided."


def format_docs(docs):
    """Format retrieved documents into a single string."""
    return "\n\n---\n\n".join(
        f"[PDF Excerpt {i+1}]:\n{doc.page_content}" for i, doc in enumerate(docs)
    )


def create_qa_chain(vectorstore):
    """Create a strict RAG chain that ONLY answers from the PDF."""

    # Step 1: Initialize Groq LLM — temperature=0 for maximum determinism
    llm = ChatGroq(
        model_name="llama-3.1-8b-instant",
        temperature=0,
    )

    # Step 2: Ultra-strict prompt that hammers home the constraint
    prompt_template = """You are a PDF-only answering machine. You have ZERO general knowledge. You know NOTHING except what is written in the PDF excerpts below.

=== ABSOLUTE RULES (NEVER BREAK THESE) ===
1. You can ONLY use information that is EXPLICITLY written in the PDF excerpts below.
2. You have NO knowledge of the world. You are NOT an AI assistant. You are a PDF reader.
3. If the PDF excerpts do NOT contain the answer, you MUST respond with EXACTLY: "{not_in_pdf}"
4. Do NOT infer, guess, elaborate, or add ANY information not in the excerpts.
5. Do NOT use phrases like "In general...", "Typically...", "Usually...", "It is known that..." — these indicate outside knowledge which is FORBIDDEN.
6. Every sentence in your answer must be directly traceable to a specific PDF excerpt below.
7. If the question is a greeting or casual conversation, respond with: "{not_in_pdf}"

=== PDF EXCERPTS (your ONLY source of truth) ===
{context}
=== END OF PDF EXCERPTS ===

Question: {question}

Instructions: Answer the question using ONLY the PDF excerpts above. If the excerpts do not contain relevant information to answer this question, say "{not_in_pdf}". Do not add anything from your own knowledge."""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"],
        partial_variables={"not_in_pdf": NOT_IN_PDF},
    )

    # Step 3: Create retriever — use k=4 for initial retrieval, then filter
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # Step 4: Build a custom chain with relevance checking
    class StrictPDFChain:
        """A chain wrapper that checks context relevance before answering."""

        def __init__(self, llm, prompt, vectorstore, retriever):
            self.llm = llm
            self.prompt = prompt
            self.vectorstore = vectorstore
            self.retriever = retriever
            self.chain = prompt | llm | StrOutputParser()

        def invoke(self, question):
            # Retrieve documents with similarity scores
            docs_with_scores = self.vectorstore.similarity_search_with_score(
                question, k=4
            )

            # FAISS returns L2 distance — lower = more similar
            # Filter out documents with very low relevance (high distance)
            # Typical threshold: L2 distance > 1.5 means very poor match
            RELEVANCE_THRESHOLD = 2.0

            relevant_docs = [
                (doc, score) for doc, score in docs_with_scores
                if score < RELEVANCE_THRESHOLD
            ]

            # If no documents are relevant enough, refuse to answer
            if not relevant_docs:
                return NOT_IN_PDF

            # Use only the top relevant documents
            docs = [doc for doc, score in relevant_docs[:3]]
            context = format_docs(docs)

            # Run the chain
            answer = self.chain.invoke({
                "context": context,
                "question": question,
            })

            # Post-processing: catch common LLM hallucination patterns
            # If the LLM ignores our prompt and answers anyway with hedging language
            hallucination_markers = [
                "in general",
                "typically",
                "it is widely known",
                "as we all know",
                "it is common knowledge",
                "outside the pdf",
                "based on my knowledge",
                "from my training",
                "i can tell you that",
                "while the pdf doesn't",
                "although the document doesn't",
                "even though the pdf",
                "beyond what's in the pdf",
                "in addition to what the pdf says",
            ]

            answer_lower = answer.lower()
            for marker in hallucination_markers:
                if marker in answer_lower:
                    return NOT_IN_PDF

            return answer

        def invoke_with_sources(self, question):
            """Same as invoke but also returns the source documents used."""
            docs_with_scores = self.vectorstore.similarity_search_with_score(
                question, k=4
            )

            RELEVANCE_THRESHOLD = 2.0
            relevant_docs = [
                (doc, score) for doc, score in docs_with_scores
                if score < RELEVANCE_THRESHOLD
            ]

            if not relevant_docs:
                return NOT_IN_PDF, []

            docs = [doc for doc, score in relevant_docs[:3]]
            context = format_docs(docs)

            answer = self.chain.invoke({
                "context": context,
                "question": question,
            })

            answer_lower = answer.lower()
            hallucination_markers = [
                "in general", "typically", "it is widely known",
                "as we all know", "it is common knowledge",
                "outside the pdf", "based on my knowledge",
                "from my training", "i can tell you that",
                "while the pdf doesn't", "although the document doesn't",
                "even though the pdf", "beyond what's in the pdf",
                "in addition to what the pdf says",
            ]

            for marker in hallucination_markers:
                if marker in answer_lower:
                    return NOT_IN_PDF, []

            return answer, docs

    strict_chain = StrictPDFChain(llm, prompt, vectorstore, retriever)

    return strict_chain, retriever


# Quick test
if __name__ == "__main__":
    from ingest import load_and_split_pdf
    from vectorstore import create_vectorstore
    import sys

    if len(sys.argv) < 2:
        print("Usage: python qa_chain.py <path_to_pdf>")
        sys.exit(1)

    chunks = load_and_split_pdf(sys.argv[1])
    vs = create_vectorstore(chunks)
    chain, retriever = create_qa_chain(vs)

    # Test question
    question = "What is a dictionary in Python?"
    print(f"\nQuestion: {question}")
    answer = chain.invoke(question)
    print(f"Answer: {answer}")
