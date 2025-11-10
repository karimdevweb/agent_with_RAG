
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

model = init_chat_model("google_genai:gemini-2.5-flash")


@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information to help answer a query."""
    retrieved_docs = vectordb.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs



tools = [retrieve_context]
# If desired, specify custom instructions
prompt = (
    "You have access to a tool that retrieves context from a vector db. "
    "Use the tool to help answer user queries."
)
agent = create_agent(model, tools, system_prompt=prompt)


query = (
    "give me a short summary of the document\n"
    "give me the second receipe you get from the document\n\n",
    "could you give back what are the instruction for that receipe, be concise\n"
)

for event in agent.stream(
    {"messages": [{"role": "user", "content": query}]},
    stream_mode="values",
):
    event["messages"][-1].pretty_print()



