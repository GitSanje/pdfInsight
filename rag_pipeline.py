import requests 
from langchain_community.document_loaders import PyPDFLoader
import tempfile


#py-spy record -o profile.speedscope.json -f speedscope python rag_pipeline.py
# https://www.speedscope.app/


import cProfile
import pstats

def get_text_from_pdf(file_path=None, url=None):
    if url:
        response = requests.get(url)
        pdf_bytes = response.content
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(pdf_bytes)
            file_path = temp_pdf.name
    loader = PyPDFLoader(file_path)
    docs = loader.load_and_split()
    return docs
docs = get_text_from_pdf(url="https://arxiv.org/pdf/2510.18234")
pages = [ doc.page_content for doc in docs ]


from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore

from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key="AIzaSyDEckSvtc3k_d0KgXyPgsvC1nUUjYc7xBk",
            
        )



splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=150,
        )

text = "".join(pages)
docs = splitter.create_documents([text])
print(len(docs))
class MemoryStore():
    def __init__(self, embeddings):
        self.store = InMemoryVectorStore(embedding=embeddings)

    def add_documents(self, documents):
      
        return  self.store.add_documents(documents)

    def similarity_search(self, query, k=4):
        return self.store.similarity_search(query, k)
    

memory_store = MemoryStore(embeddings)

ids = memory_store.add_documents(docs)
print(f"Added {len(ids)} documents to memory store")

def print_state(state, node_name):
    print(f"\n=== {node_name} ===")
    for i, m in enumerate(state["messages"]):
        print(f"[{i}] {m.__class__.__name__}: {m.content}")
        if getattr(m, "tool_calls", None):
            print("   tool_calls:", m.tool_calls)
    print("==============\n")


import asyncio
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool

class LLMProvider:
    def __init__(self, api_key: str):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0,
            api_key=api_key,
        )
    def base(self):
        return self.llm
    
    def with_tools(self, tools):
        return self.llm.bind_tools(tools) 
    
class RetrievalTool:
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
        self.tool = self._build_tool()

    def _build_tool(self):
        
        @tool(description="Retrieve documents from the vector store based on a user query.")
        async def retrieve(query: str):
            #similarity_search() is blocking CPU-bound code
            #Runs the function in a thread pool
            #Frees the event loop
            docs = await asyncio.to_thread(
                   self.vectorstore.similarity_search, query, 2
              )
            print(f"Retrieved {docs} documents for query: {query}")

            serialized = "\n".join(
                f"Source: {d.metadata.get('source')}\n{d.page_content}"
                for d in docs
            )
            return serialized

        return retrieve


from typing import TypedDict, Annotated, List
from langgraph.graph.message import  add_messages
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import (
    SystemMessage, AIMessage, ToolMessage, HumanMessage
)
from langgraph.prebuilt import ToolNode


class GraphState(TypedDict):
      messages: Annotated[List[BaseMessage], add_messages]
    

class RAGNodes:
    def __init__(self, llm_provider : LLMProvider, retrieve_tool:RetrievalTool):
        self.llm_provider = llm_provider
        self.retrieve_tool = retrieve_tool

    async def query_or_respond(self, state: GraphState):
        
        llm = self.llm_provider.with_tools([self.retrieve_tool.tool])
        messages = list(state["messages"])
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages.insert(0, SystemMessage(
                "Use the retrieve tool if external information is required."
            ))
        print_state(state, "query_or_respond")
        print( "state in query_or_respond: ",state )
        response =  await llm.ainvoke(messages)
        print("Response llm: ", response)
        return {"messages": [response]}
    
    async def generate(self, state: GraphState):
        tool_messages = []
        print("Generating with retrieved context...")
        print_state(state, "generate")
        print( "state in generate: ",state )
        for msg in reversed(state["messages"]):
            if isinstance(msg, ToolMessage):
                tool_messages.append(msg)
            else:
                break

        docs_context = "\n".join(m.content for m in reversed(tool_messages))

        system = SystemMessage(f"""
            You are a helpful assistant.
            Use the retrieved context if relevant.

            Context:
            ---------
            {docs_context}
            ---------
            """)

        convo = [
            m for m in state["messages"]
            if isinstance(m, (HumanMessage, AIMessage))
            and not getattr(m, "tool_calls", None)
        ]

        response =  await self.llm_provider.base().ainvoke(
            [system, *convo]
        )

        return {"messages": [response]}
    
    


    


class RAGGraphBuilder:
    def __init__(self, nodes, retrieve_tool, checkpointer):
        self.nodes = nodes
        self.retrieve_tool = retrieve_tool
        self.checkpointer = checkpointer
    
    @staticmethod
    def tools_condition(state: GraphState):
        last = state["messages"][-1]
        print_state(state, "tools_condition")
        print( "state in tools_condition: ",state )
        if getattr(last, "tool_calls", None):
            return "tools"
        return END

    def build(self):
        graph = StateGraph(GraphState)

        graph.add_node("queryOrRespond", self.nodes.query_or_respond)
        graph.add_node("tools", ToolNode([self.retrieve_tool.tool]))
        graph.add_node("generate", self.nodes.generate)

        graph.add_edge(START, "queryOrRespond")
        graph.add_conditional_edges("queryOrRespond", self.tools_condition, {
            "tools": "tools",
            END: END
        })
        graph.add_edge("tools", "generate")
        graph.add_edge("generate", END)

        return graph.compile(checkpointer=self.checkpointer)

from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig


checkpointer = MemorySaver()
class RAGPipeline:
    def __init__(self, api_key, checkpointer,vectorstore):
        self.vectorstore =vectorstore
        self.llm_provider = LLMProvider(api_key)
        self.checkpointer = checkpointer

    async def run(self, query, thread_id, user_id, pdf_id):

        retrieve_tool = RetrievalTool(self.vectorstore)
        nodes = RAGNodes(self.llm_provider, retrieve_tool)

        graph = RAGGraphBuilder(
            nodes, retrieve_tool, self.checkpointer
        ).build()
       

        print("Running RAG graph...", graph)
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

        result =  await graph.ainvoke(
            {"messages": [HumanMessage(query)]},
            config
        )
            
        print(result)

        return next(
            (m.content for m in reversed(result["messages"])
             if isinstance(m, AIMessage)),
            ""
        )



async def main():

    pipeline = RAGPipeline(
        api_key="AIzaSyDEckSvtc3k_d0KgXyPgsvC1nUUjYc7xBk",
        checkpointer=checkpointer,
        vectorstore=memory_store
    )


    result = await pipeline.run("What is the main contribution of the paper?", thread_id="thread1", user_id="user1", pdf_id="pdf1" )
    print(f"**Final Result:**\n\n{result}")



if __name__ == "__main__":
    # profiler = cProfile.Profile()
    # profiler.runcall(main)
    # stats = pstats.Stats(profiler)
    # stats.sort_stats("cumulative").print_stats(30)
    from pycallgraph2 import PyCallGraph
    from pycallgraph2.output import GraphvizOutput
    from viztracer import VizTracer

    with cProfile.Profile() as profiler:
        asyncio.run(main())
    stats = pstats.Stats(profiler)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.dump_stats("profiling_results.prof")

    # graphviz = GraphvizOutput()
    # graphviz.output_file = 'callgraph.png'
    # tracer = VizTracer(
    #     log_async=True,       # track async tasks
    #     log_print=False,      # optional: track print statements
    #     output_file="trace.json"
    # )

    # tracer.start()
    # asyncio.run(main())
    # tracer.stop()
    # tracer.save()  # Produces trace.json
    # print("Trace saved to trace.json")


    # with PyCallGraph(output=graphviz):
    #     main()


