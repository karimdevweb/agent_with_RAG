
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
os.environ["GOOGLE_API_KEY"] = "A exemple api key ugw"

model = init_chat_model("google_genai:gemini-2.0-flash-lite")


@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information to help answer a query."""
    retrieved_docs = vectordb.similarity_search(query, k=3)
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




#  ------------------------------ from here the evaluation of the RAG agent -----------------------------------
# import ragas libraries
from datasets import Dataset
from ragas import evaluate
from ragas import SingleTurnSample
from ragas.metrics import ResponseRelevancy, Faithfulness, ContextPrecision
import asyncio
import re


# LLM d'évaluation = Gemini
evaluator_llm = init_chat_model("google_genai:gemini-2.0-flash-lite")
# Embeddings d'évaluation = OllamaEmbeddings
evaluator_embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")



#  build the truth basis and the questions
ground_truths = [
    "this documents describe how to prepare some receipes, it gives back the ingredients and instructions to follow for backing delicious desserts.",
    "the second recipe elaborate the beef in beer dish",
    "Raw beef, roll it in the flour...fry 4-­‐5 pieces at a time in a hot saucepan to cut the onions into quarters...fry in	the	same frying	pan	as the beef...to soak up the juices.",
    "thai green curry is the fourth recipe",
    "yes, the sixth recipe is vegetarian and the main ingredient is rice"
]
query = (
    "give me a short summary of the document\n",
    "give me the second recipe you get from the document\n\n",
    "could you give back what are the instruction for that recipe, be concise\n",
    "in which position comes the Thai green curry recipe ?",
    "Is there any vegetarian recipe containing rice ?"
)

# now, we have to catch the answwers and build the context
answers = []
contexts = []

for q in query:
    # Appelle ton agent RAG
    result = agent.invoke({"messages": [{"role": "user", "content": q}]})

    # Récupère la réponse du modèle en string
    raw_content = result["messages"][-1].content
    if isinstance(raw_content, list):
        answer_text = "".join([part.get("text", "") for part in raw_content])
    else:
        answer_text = str(raw_content)
    # answer_text = re.sub(r"^```json|```$", "", answer_text.strip(), flags=re.MULTILINE).strip()
    answers.append(answer_text)

    # Utilise ton retriever pour obtenir les contextes
    docs = vectordb.similarity_search(q, k=3)
    contexts.append([doc.page_content for doc in docs])




# initialiser les métriques RAGAS
scorer_relevancy = ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings)
scorer_faith = Faithfulness(llm=evaluator_llm)
scorer_context = ContextPrecision(llm=evaluator_llm)



# preparer ragas - pour le context et query
data = {
    "question": [query],
    "contexts": [contexts],
    "answer": [answers],
    "ground_truth": ground_truths
}

# créer les samples et scorer
async def evaluate_samples():
    for i, q in enumerate(query):
        sample = SingleTurnSample(
            user_input=q,
            response=answers[i],
            retrieved_contexts=contexts[i],
            ground_truth=ground_truths[i]
        )

        score_rel = await scorer_relevancy.single_turn_ascore(sample)
        score_fai = await scorer_faith.single_turn_ascore(sample)
        score_ctx = await scorer_context.single_turn_ascore(sample)

        print(f"\nQuestion {i+1}: {q}")
        print(f"Réponse: {answers[i]}")
        print(f"Scores -> Relevancy: {score_rel}, Faithfulness: {score_fai}, ContextPrecision: {score_ctx}")

# exécuter l’évaluation
# asyncio.run(evaluate_samples())

print("------------------------- question réponses ----------------------------------------------")
print("longueur des questions:" , len(query))
print("longueur des réponses:" , len(answers))
for qst , res in zip(query, answers):
    print("->",  qst)
    print("=>" , res)
    print("-----------------------------------------------------------------------")


