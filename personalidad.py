from fastapi import FastAPI
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate 
from langchain.memory import ConversationBufferWindowMemory
from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools.render import format_tool_to_openai_function
from langchain.vectorstores import faiss
from pydantic import BaseModel, ValidationError
from dotenv import load_dotenv, find_dotenv
from langserve import add_routes


load_dotenv()
vectorstore = faiss.from_texts(
    ["Lay ayuda a los usuarios de Tabantaj"], embedding=OpenAIEmbeddings
)
retriever=vectorstore.as_retriever()


@tool
def get_Lay_thoughts(query:str) -> list:
    """Nos devuelve los pensamientos de Lay a cerca del tema"""
    return retriever.get_relevant_documents(query)

tools=[get_Lay_thoughts]

prompt= ChatPromptTemplate.from_messages(
    [
        ("system", "Eres Lay, el asistente personal de Tabantaj, el cual guia al usuario a traves de las aplicaciones"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="asistente_virtual"),
    ]
)

llm=ChatOpenAI()

llm_with_tools= llm.bind(functions=[format_tool_to_openai_functions(t) for t in tools])

agent= (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_functions(
            x["intermediate_steps"]
        ),
    }
        prompt
        llm_with_tools
        OpenAIFunctionsAgentOutputParser(),
) 

agent_executor= AgentExecutor(agent=agent, tools=tools)

app=FastAPI(
    title="LangChain Server",
    version= "1.0",
    description="Spin up a simple api server using Langchain's Runnable interfaces",
)


class Input(BaseModel):
    input: str
    
class Output(BaseModel):
    output: str
    

add_routes(app,agent_executor, input_type=Input, output_type=Output)

if __name__=="_main_"
    import uvicorn
    
    uvicorn.run(app, host="localhost", port=8000)