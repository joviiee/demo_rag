import os 
from dotenv import load_dotenv
from typing import Sequence
from typing_extensions import Annotated, TypedDict,List

from langchain.chat_models import init_chat_model
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector
from langchain_core.tools import tool
from langchain_core.documents import Document
from langgraph.graph import StateGraph,START
from langgraph.graph.message import add_messages

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import SystemMessage,BaseMessage

from db_config import DB_CONNECTION_STRING

load_dotenv()

class State(TypedDict):
    messages:Annotated[Sequence[BaseMessage],add_messages]
    context:List[Document]
    answer:str
    doc:str
    num_results:int

class MutableState:
    def __init__(self, initial_state: dict):
        self.state = initial_state

llm = init_chat_model("gemini-1.5-pro", model_provider="google_genai")
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
memory = MemorySaver()

graph_builder = StateGraph(State)


def retrieve(state:State):
    retriever = PGVector(
            embedding=embedding_model,
            collection_name= state["doc"],
            connection=DB_CONNECTION_STRING
        )
    retrived_docs = retriever.similarity_search(query=state[], k=state_ref.state["num_results"])
    serialised = "\n\n".join(
        (f"Source : {doc.metadata}\nContent:{doc.page_content}")
        for doc in retrived_docs
    )
    return serialised, retrived_docs


reference_state = MutableState({"doc":"data/agentic_ai.pdf" , "num_results":2})

retireve_tool = make_retrieve_tool(reference_state)

def copy_state(state:State):
    reference_state.state["doc"] = state["doc"]
    reference_state.state["num_results"] = state["num_results"]
    return state



