from langchain.chains import GraphCypherQAChain

from llm import llm
from graph import graph
from langchain.tools import tool
from langchain.prompts.prompt import PromptTemplate


CYPHER_GENERATION_TEMPLATE = """
You are an expert Neo4j Developer translating user questions into Cypher to answer questions about movies and provide recommendations.
Convert the user's question based on the schema.

Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.

Fine Tuning:

For movie titles that begin with "The", move "the" to the end. For example "The 39 Steps" becomes "39 Steps, The" or "the matrix" becomes "Matrix, The".


Schema:
{schema}

Question:
{question}

Cypher Query:
"""

cypher_prompt = PromptTemplate.from_template(CYPHER_GENERATION_TEMPLATE)

qa_chain = GraphCypherQAChain.from_llm(
    llm,          # (1)
    graph=graph,  # (2)
    verbose=True,
    cypher_prompt=cypher_prompt
)


@tool()
def cypher_qa(query: str) -> str:
    """Provides information about Movies including their Actors, Directors and User reviews."""
    response = qa_chain.invoke({"query": query})
    return response['result']
