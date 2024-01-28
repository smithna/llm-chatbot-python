import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx
import openai
from tools.vector import search
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.tools.render import format_tool_to_openai_function
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.prompts import MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.agents import AgentExecutor
from langchain.memory import ConversationBufferMemory
from tools.cypher import cypher_qa
from langchain_community.chat_message_histories import Neo4jChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

openai.api_key = st.secrets['OPENAI_API_KEY']

tools = [search, cypher_qa]

functions = [format_tool_to_openai_function(f) for f in tools]
model = ChatOpenAI(temperature=0).bind(functions=functions)
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a movie expert providing information about movies.
                  Be as helpful as possible and return as much information as possible.
                  Do not answer any questions that do not relate to movies, actors or directors.
                  Always call one of your functions.
                  Do not answer any questions using your pre-trained knowledge, only use the information provided in the context."""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])


agent_chain = RunnablePassthrough.assign(
    agent_scratchpad= lambda x: format_to_openai_functions(x["intermediate_steps"])
) | prompt | model | OpenAIFunctionsAgentOutputParser()

memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")

agent_executor = AgentExecutor(agent=agent_chain, tools=tools, verbose=True, memory=memory)

history_chain = RunnableWithMessageHistory(
    agent_executor,
    lambda session_id:  Neo4jChatMessageHistory(
        url=st.secrets["NEO4J_URI"],             # (2)
        username=st.secrets["NEO4J_USERNAME"],   # (3)
        password=st.secrets["NEO4J_PASSWORD"],   # (4)
        session_id=session_id,),
    input_messages_key="input",
    history_messages_key="chat_history",
)


def generate_response(prompt):
    """
    Create a handler that calls the Conversational agent
    and returns a response to be rendered in the UI
    """
    response = history_chain.invoke({"input": prompt},
                                    config={"configurable": {"session_id": get_script_run_ctx().session_id}})

    return response['output']


