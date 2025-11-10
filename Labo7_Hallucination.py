from pypdfium2 import PdfDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter
#  chroma , that will play DB role
from langchain_chroma import Chroma
# from langchain.embeddings import ollamaEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama






#  set the model // no memory for local treatment 
embeddings = OllamaEmbeddings(
    model="nomic-embed-text:latest",
)



#  read a pdf, i downloaded a pdf to the folder
reader = PdfDocument("Agent_With_RAG/Dessert_recipes.pdf") 
# text = "".join(page.extract_text() for page in reader.pages)
text = ""
for page in reader:
    text += page.get_textpage().get_text_range()
# split the text into chunks of 500 token, let the chunks overlap one another over 100 tokens
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200) 
chunks = splitter.split_text(text)




#  index the chuns , and reate a db
vectordb = Chroma.from_texts(chunks, embedding=embeddings)

# ------------ till here the db is complete,  now we have to build the agent to get the data

#  apparentllmy we have to construct a tool first , what does that mean , no idea !!!
from langchain.tools import tool
#  then call the agent 
from langchain.agents import create_agent


# call a chat model , the localy ones are too havy, i will call gemeni

import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
#  expose api key in env, not working if it is a variable varkey=load_dotenv()
os.environ["GOOGLE_API_KEY"] = "AIfdf exemple api key TJxD8F3ugw"

model = init_chat_model("google_genai:gemini-2.5-flash-lite")
# model = ChatOllama(model="mistral:latest", temperature=0)

# Forcer l’ancrage : 
query_FL = """
Summary me the doc please,
base your response on the doc please.
"""
# Ajouter les citations : 
query_Citation = """
Summary me the doc please,
include some extract
"""

# ancienne réponse: 
query_Verif = f"""
is this response faithful ? 
The document contains a variety of recipes, including:

*   Chili con Carne
*   Beef in beer
*   Pad Thai Chicken
*   Thai Green Curry
*   Pineapple Chicken
*   Vegetarian Rice
*   Pad Thai Chicken
*   Thai Green Curry
*   Pineapple Chicken
*   Vegetarian Rice
*   Omelette
*   Egg Fried Rice
*   Salmon in the Oven
*   Vegetarian Rice
*   Omelette
*   Egg Fried Rice
*   Salmon in the Oven
*   Omelette
*   Egg Fried Rice
*   Salmon in the Oven
*   Salmon in the Oven
*   Basic Pasta
*   Carbonara al Funghi
*   Simple Spaghetti Bolognese
*   Cucumber Salad
*   Goat Cheese and Beetroot Salad

Some recipes, like Chili con Carne and Carbonara al Funghi, include detailed ingredients and methods."""


#  just change here the query 
query = query_Verif


# trying to get an hybride saerch 
from rank_bm25 import BM25Okapi

tokenized_chunks = [chunk.split() for chunk in chunks]  # simple tokenization
bm25 = BM25Okapi(tokenized_chunks)


@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information to help answer a query."""

    # get only the top 3 answers
    retrieved_docs_with_score = vectordb.similarity_search_with_score(query, k=10)

    # BM25 : score lexical
    query_tokens = query.split()
    bm25_scores = bm25.get_scores(query_tokens)

    # Combiner scores vectorielle + BM25
    combined = []
    for i, (doc, vector_score) in enumerate(retrieved_docs_with_score):
        combined_score = 0.7 * vector_score + 0.3 * bm25_scores[i]  # poids simple
        combined.append((doc, combined_score))

    # Trier et garder top 3
    combined.sort(key=lambda x: x[1], reverse=True)
    top3_docs = [doc for doc, score in combined[:3]]


    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in top3_docs
    )
    return serialized, top3_docs



tools = [retrieve_context]
# If desired, specify custom instructions
prompt = (
    "You are an assistant with access to a retrieval tool named `retrieve_context`. "
    "Always call this tool automatically when you receive a question from the user. "
    "Use it with the user's query string to find relevant context from the vector database, "
    "then answer based on that retrieved context. Do not explain the tool usage."
)

agent = create_agent(model, tools, system_prompt=prompt)



# get one response for one question
for event in agent.stream(
    {"messages": [{"role": "user", "content": query}]},
    stream_mode="values",
):
    event["messages"][-1].pretty_print()




