import os 
from dotenv import load_dotenv
from typing import Sequence
from typing_extensions import Annotated, TypedDict

from langchain.chat_models import init_chat_model
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector
from langchain_core.tools import tool
from langgraph.graph import StateGraph,START
from langgraph.graph.message import add_messages
from langchain import hub

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition, create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import SystemMessage,BaseMessage,HumanMessage

from db_config import DB_CONNECTION_STRING

load_dotenv()

class State(TypedDict):
    messages:Annotated[Sequence[BaseMessage],add_messages]
    doc:str
    num_results:int
    temperature:float


llm = init_chat_model("gemini-1.5-pro", model_provider="google_genai")
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
memory = MemorySaver()
prompt_template = hub.pull("rlm/rag-prompt")

graph_builder = StateGraph(State)

def check_for_tool_calls(state:State):
    last_message = state["messages"][-1]
    return (hasattr(last_message, "tool_calls") and bool(last_message.tool_calls))

@tool(response_format="content_and_artifact", description="Retrive data from vector store")
def retrieve(query:str):
    retriever = PGVector(
            embeddings=embedding_model,
            collection_name="embeddings",
            connection=DB_CONNECTION_STRING
        )
    retrieved_docs = retriever.similarity_search(query = query, k = 3)
    print(retrieved_docs)
    serialised = "\n\n".join(
        (f"Source : {doc.metadata}\nContent:{doc.page_content}")
        for doc in retrieved_docs
    )
    return serialised, retrieved_docs
    

tools = ToolNode([retrieve])
llm_with_tools = llm.bind_tools(tools = [retrieve], tool_choice="auto")

def query_or_respond(state:State):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages":[response]}

def generate(state:State):
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break

    tool_messages = recent_tool_messages[::-1]

    search_results = "\n\n".join(result.content for result in tool_messages)

    system_message_content = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        f"{search_results}"
    )

    conversational_messages = [
        message 
        for message in state["messages"]
        if message.type in ("human", "system") or (message.type == "ai" and not message.tool_calls)
    ]
    prompt = [SystemMessage(system_message_content)] + conversational_messages
    response = llm.invoke(prompt)

    return {"messages" : [response]}


graph_builder.add_node(query_or_respond)
graph_builder.add_node(tools)
graph_builder.add_node(generate)

graph_builder.set_entry_point("query_or_respond")
graph_builder.add_conditional_edges(
    "query_or_respond",
    check_for_tool_calls,
    {False:END, True:"tools"}
)
graph_builder.add_edge("tools","generate")
graph_builder.add_edge("generate", END)

graph = graph_builder.compile(checkpointer = memory)

config = {"configurable": {"thread_id": "abc123"}}

while True:
    input_message = input("Enter the prompt : ")

    for step in graph.stream(
        {"messages": [{"role": "user", "content": input_message}]},
        stream_mode="values",
        config=config
    ):
        step["messages"][-1].pretty_print()

