import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain.tools.retriever import create_retriever_tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver



memory = MemorySaver()
load_dotenv()

def get_chatbot_ai_output(input_query) :
        model = AzureChatOpenAI(
            model="gpt-4o-mini",
            api_key=os.getenv('AZURE_OPENAI_API_KEY'),
            azure_endpoint= os.getenv('AZURE_OPENAI_ENDPOINT'),
            azure_deployment=os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME'),
            openai_api_version=os.getenv('AZURE_OPENAI_API_VERSION'),
            temperature=0.3
        )

        leave_policy_tool = get_retriever_tool(os.path.join(os.getcwd(), "Leave Policy.pdf"),"policy_retriever","Searches and returns queries from leave policy.")
        weather_policy_tool = get_retriever_tool(os.path.join(os.getcwd(), "AIWFB.pdf"),"weather_retriever","Searches and returns queries from weather document.")
        tools = [leave_policy_tool,weather_policy_tool]

        agent_executor = create_react_agent(model, tools, checkpointer=memory)

        config = {"configurable": {"thread_id": "abc123"}}
        response = agent_executor.invoke({"messages": [HumanMessage(content=input_query)]}, config=config)
        ai_messages = [message.content for message in response["messages"] if isinstance(message, AIMessage)]
        return list(filter(lambda content : content != "", ai_messages))
        

def get_retriever_tool(filepath, retriever_name, retriever_desc ):
        loader = PyPDFLoader(filepath)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        vectorstore = InMemoryVectorStore.from_documents(
            documents=splits, embedding=AzureOpenAIEmbeddings(
                model="text-embedding-3-large",
            )
        )
        retriever = vectorstore.as_retriever()
        return create_retriever_tool(
            retriever,
            retriever_name,
            retriever_desc,
        )

