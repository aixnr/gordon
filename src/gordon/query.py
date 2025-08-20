import argparse
from langchain_community.vectorstores import FAISS
from rich.console import Console
from rich.markdown import Markdown
from rich.pretty import pprint
from prompt_toolkit import prompt as prpt
import tiktoken

from .loadmodel import embeddings
from .graph import return_graph


def main():
    parser = argparse.ArgumentParser(
        description="Run your RAG program with FAISS vector store and LangChain."
    )
    parser.add_argument(
        "--index-path",
        type=str,
        default="faiss_index",
        help="Path to the FAISS index folder (default: %(default)s)",
    )
    parser.add_argument(
        "--print-context",
        action="store_true",
        help="Whether to pprint the retrieved context documents.",
    )
    args = parser.parse_args()

    vectordb = FAISS.load_local(
        args.index_path,
        embeddings,
        allow_dangerous_deserialization=True
    )

    graph = return_graph(vectordb=vectordb)

    console = Console()
    encoding = tiktoken.encoding_for_model("gpt-4o")
    while True:
        console.print()
        try:
            query = prpt("Your question (or type 'exit'): ")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[bold red]Exiting...[/bold red]")
            break
        console.print()
        if query.lower() == "exit":
            break

        result = graph.invoke({"question": query})

        if args.print_context:
            pprint(result["context"])

        output_text = result["answer"]
        output_token = len(encoding.encode(output_text))
        console.print(Markdown(output_text))
        console.print(Markdown("---"))
        console.print(f"[bold green]Output Tokens:[/bold green] {output_token}")
        console.print(Markdown("---"))


if __name__ == "__main__":
    main()
