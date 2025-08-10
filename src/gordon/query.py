import argparse
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from rich.console import Console
from rich.markdown import Markdown
from rich.pretty import pprint
from rich.prompt import Prompt
from .loadmodel import embeddings, llm


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

    template_prompt = """
    Use the following pieces of context to answer the question at the end.
    If the context doesn't provide enough information, just say that you don't know, don't try to make up an answer.
    Include as much details as possible.
    {context}
    Question: {question}
    Helpful Answer:
    """
    prompt = PromptTemplate.from_template(template_prompt)

    class State(TypedDict):
        question: str
        context: List[Document]
        answer: str

    def retrieve(state: State):
        retrieved_docs = vectordb.similarity_search(state["question"])
        return {"context": retrieved_docs}

    def generate(state: State):
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = prompt.invoke({"question": state["question"], "context": docs_content})
        response = llm.invoke(messages)
        return {"answer": response.content}

    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()

    console = Console()
    while True:
        console.print()
        try:
            query = Prompt.ask("[bold]Your question (or type 'exit')[/bold]")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[bold red]Exiting...[/bold red]")
            break
        console.print()
        if query.lower() == "exit":
            break
        result = graph.invoke({"question": query})
        if args.print_context:
            pprint(result["context"])
        md = Markdown(result["answer"])
        console.print(md)


if __name__ == "__main__":
    main()
