import os
import sys
import argparse

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from .loadmodel import embeddings

def load_papers(directory: str):
    docs = []
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(directory, filename))
            docs.extend(loader.load())
    return docs


def main():
    parser = argparse.ArgumentParser(description="Ingest PDFs into or add to FAISS vector store.")
    parser.add_argument(
        "--doc-dir", default="documents",
        help="Directory containing PDF documents to ingest"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=1024,
        help="Chunk size for splitting documents"
    )
    parser.add_argument(
        "--chunk-overlap", type=int, default=128,
        help="Chunk overlap size for splitting documents"
    )
    parser.add_argument(
        "--output", default="faiss_index",
        help="Directory where FAISS index is saved/loaded"
    )
    parser.add_argument(
        "--device", default="cpu",
        help="Device for embeddings model (eg. 'cpu' or 'cuda')"
    )
    args = parser.parse_args()

    print(f"[*] Loading papers from '{args.doc_dir}' ...")
    raw_docs = load_papers(args.doc_dir)
    if not raw_docs:
        print(f"[!] No PDF documents found in '{args.doc_dir}'. Exiting.")
        sys.exit(1)

    print("[*] Splitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap
    )
    docs = splitter.split_documents(raw_docs)

    index_path = args.output
    if os.path.exists(index_path):
        print(f"[*] Loading existing FAISS index from '{index_path}' and adding documents...")
        vectordb = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        vectordb.add_documents(docs)
    else:
        print("[*] Creating new FAISS index from documents...")
        vectordb = FAISS.from_documents(docs, embeddings)

    print(f"[*] Saving vector store to '{index_path}' ...")
    vectordb.save_local(index_path)
    print("[*] Done.")


if __name__ == "__main__":
    main()
