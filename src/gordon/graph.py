from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from typing_extensions import List, TypedDict
from langchain_community.vectorstores import FAISS

from .loadmodel import llm


def return_graph(vectordb: FAISS) -> CompiledStateGraph:
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

    return graph
