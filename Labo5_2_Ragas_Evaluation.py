
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
os.environ["GOOGLE_API_KEY"] = "AIzaSyBuG5ZfnCyhmGQiX8Q6aDsHbTJxD8F3ugw"

model = init_chat_model("google_genai:gemini-2.5-flash" ,     
            # response_mime_type= "application/json"
        )


@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information to help answer a query."""
    retrieved_docs = vectordb.similarity_search(query, k=5)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs



tools = [retrieve_context]
# If desired, specify custom instructions
prompt = (
   """ You are an assistant with access to a retrieval tool named `retrieve_context`.
    - Always call this tool automatically for user questions.
    - Use the retrieved context to answer the question.
    - Do not just rewrite the question; provide the actual answer.
"""
)
agent = create_agent(model, tools, system_prompt=prompt)




#  ------------------------------ from here the evaluation of the RAG agent -----------------------------------
# import ragas libraries
from datasets import Dataset
from ragas import evaluate
from ragas import SingleTurnSample
from ragas.metrics import ResponseRelevancy, Faithfulness, ContextPrecision
import asyncio
import re


# LLM d'évaluation = Gemini
evaluator_llm = init_chat_model("google_genai:gemini-2.5-flash" ,     
                        # response_mime_type= "application/json" 
                        )

# Embeddings d'évaluation = OllamaEmbeddings
evaluator_embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")


def clean_json_output(text):
    # Remove ```json ... ``` wrappers
    return re.sub(r"^```json|```$", "", text.strip(), flags=re.MULTILINE).strip()

# Single question, single answer, single context
query = "could you resume the document pdf please, the one about recipes"

result = agent.invoke({"messages": [{"role": "user", "content": query}]})

# clean the answer, otherwise ragas won't accept it
raw_content = result["messages"][-1].content
if isinstance(raw_content, list):
    raw_text = "".join([part.get("text", "") for part in raw_content])
else:
    raw_text = str(raw_content)

answer = result["messages"][-1].content[0]['text']



docs = vectordb.similarity_search(query, k=5)
context = [doc.page_content for doc in docs]
ground_truth = "the pdf contains recipes about dishes"








# initialiser les métriques RAGAS
scorer_relevancy = ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings)
scorer_faith = Faithfulness(llm=evaluator_llm)
scorer_context = ContextPrecision(llm=evaluator_llm)





# créer les samples et scorer
async def evaluate_samples():
    # Build one RAGAS sample
    sample = SingleTurnSample(
        user_input=query,
        response=answer,
        retrieved_contexts=context,
        ground_truth=ground_truth
    )

    score_rel = await scorer_relevancy.single_turn_ascore(sample)
    score_fai = await scorer_faith.single_turn_ascore(sample)
    score_ctx = await scorer_context.single_turn_ascore(sample)

    print(f"\nQuestion: {query}")
    print(f"Réponse: {answer}")
    print(f"Scores -> Relevancy: {score_rel}, Faithfulness: {score_fai}, ContextPrecision: {score_ctx}")


# exécuter l’évaluation
asyncio.run(evaluate_samples())

