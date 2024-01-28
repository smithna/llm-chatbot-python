import streamlit as st
from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from llm import llm, embeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain import hub
from langchain.tools import tool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI

neo4jvector = Neo4jVector.from_existing_index(
    embeddings,                              # (1)
    url=st.secrets["NEO4J_URI"],             # (2)
    username=st.secrets["NEO4J_USERNAME"],   # (3)
    password=st.secrets["NEO4J_PASSWORD"],   # (4)
    index_name="moviePlots",                 # (5)
    node_label="Movie",                      # (6)
    text_node_property="plot",               # (7)
    embedding_node_property="plotEmbedding", # (8)
    retrieval_query="""
RETURN
    node.plot AS text,
    score,
    {
        title: node.title,
        directors: [ (person)-[:DIRECTED]->(node) | person.name ],
        actors: [ (person)-[r:ACTED_IN]->(node) | [person.name, r.role] ],
        tmdbId: node.tmdbId,
        source: 'https://www.themoviedb.org/movie/'+ node.tmdbId
    } AS metadata
"""
)

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a movie expert providing information about movies.
                  Be as helpful as possible and return as much information as possible.
                  Do not answer any questions that do not relate to movies, actors or directors.
                  Always the title of the movie in your answer.
                  Answer any questions based solely on the context below:
                  <context>
                  {context}
                  </context>"""),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("user", "{input}"),
])

movie_info_template= "{page_content}, title: {title}, directors: {directors}, actors: {actors}, tmdbId:{tmdbId}, source:{source}"
document_prompt = PromptTemplate(input_variables=["page_content", "title", "directors", "actors", "tmdbId", "source"],
                                 template=movie_info_template
                                 )


retriever = neo4jvector.as_retriever()

retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

combine_docs_chain = create_stuff_documents_chain(
    ChatOpenAI(temperature=0), prompt, document_prompt=document_prompt

)
retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)


@tool(return_direct=True)
def search(query: str) -> str:
    """Provides information about movie plots using Vector Search."""
    response = retrieval_chain.invoke({"input": query})
    return response['answer']
    #response = retriever.invoke(query)
    #return response
